import asyncio
import websockets
import json
import os
import base64
import pyaudio
import threading
import time
from collections import deque
from dotenv import load_dotenv

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 512

stop_event = threading.Event()
send_queue = asyncio.Queue()


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Nagrywanie... Naciśnij Ctrl+C aby zatrzymać")

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            asyncio.run_coroutine_threadsafe(send_queue.put(data), record_audio.loop)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Błąd nagrywania: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        stop_event.set()


async def send_audio(websocket):
    while not stop_event.is_set():
        data = await send_queue.get()
        audio_data = base64.b64encode(data).decode("utf-8")
        message = {"type": "input_audio_buffer.append", "audio": audio_data}
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            print("Błąd wysyłania audio:", e)


class AudioPlayer:
    def __init__(self, min_buffer_duration=2.0):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
        self.buffer = bytearray()
        self.min_buffer_size = int(RATE * min_buffer_duration * CHANNELS * 2)  # 2 sekundy bufora
        self.playing = False
        self.lock = threading.Lock()

    def add_audio(self, audio_data):
        with self.lock:
            self.buffer.extend(audio_data)

            # Rozpocznij odtwarzanie jeśli mamy wystarczająco duży bufor
            if not self.playing and len(self.buffer) >= self.min_buffer_size:
                self.playing = True
                threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        try:
            while self.playing and not stop_event.is_set():
                with self.lock:
                    # Sprawdź czy mamy wystarczająco danych do odtworzenia
                    if len(self.buffer) < CHUNK * 2:  # 2 bajty na próbkę dla paInt16
                        time.sleep(0.01)
                        continue

                    # Pobierz dane z bufora
                    data = bytes(self.buffer[:CHUNK * 2])
                    self.buffer = self.buffer[CHUNK * 2:]

                # Odtwórz dane
                try:
                    self.stream.write(data)
                except Exception as e:
                    print(f"Błąd odtwarzania: {e}")
                    break
        except Exception as e:
            print(f"Błąd w wątku odtwarzania: {e}")
        finally:
            self.playing = False

    def close(self):
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()


async def openai_realtime():
    try:
        with open("oferta.txt", "r", encoding="utf-8") as f:
            oferta_pracy = f.read().strip()
    except FileNotFoundError:
        print("Brak pliku oferta.txt – utwórz go i wklej ofertę pracy.")
        return

    instructions_pl = (
        "Jesteś rekruterem prowadzącym rozmowę kwalifikacyjną. "
        "Rozmowa ma dotyczyć WYŁĄCZNIE poniższej oferty pracy. "
        "Nie wolno Ci odpowiadać na pytania niezwiązane z tą ofertą. "
        "Jeśli użytkownik pyta o coś innego, powiedz krótko: "
        "'Odpowiadam tylko na pytania dotyczące tej oferty pracy.'\n\n"
        f"OFERTA PRACY:\n{oferta_pracy}"
    )

    # Inicjalizacja odtwarzacza audio z 2-sekundowym buforem
    audio_player = AudioPlayer(min_buffer_duration=2.0)

    async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            additional_headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1"
            }
    ) as websocket:

        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions_pl
            }
        }))

        record_audio.loop = asyncio.get_event_loop()
        record_thread = threading.Thread(target=record_audio)
        record_thread.daemon = True
        record_thread.start()

        asyncio.create_task(send_audio(websocket))

        await asyncio.sleep(1)
        print("Rozmowa rozpoczęta. Mów do mikrofonu...")

        try:
            async for message in websocket:
                data = json.loads(message)

                if data['type'] == 'response.audio.delta':
                    audio_bytes = base64.b64decode(data['delta'])
                    audio_player.add_audio(audio_bytes)

                elif data['type'] == 'response.content_part.done':
                    if 'transcript' in data['part']:
                        print(f"Asystent: {data['part']['transcript']}")

                elif data['type'] == 'error':
                    print(f"Błąd: {data['error']}")

                elif data['type'] == 'session.updated':
                    print("Sesja zainicjalizowana")

                elif data['type'] == 'response.created':
                    print("Otrzymywanie odpowiedzi...")

        except Exception as e:
            print(f"Błąd połączenia: {e}")
        finally:
            stop_event.set()
            audio_player.close()
            try:
                record_thread.join(timeout=1.0)
            except:
                pass


if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Ustaw zmienną środowiskową OPENAI_API_KEY")
        exit(1)

    try:
        asyncio.run(openai_realtime())
    except KeyboardInterrupt:
        print("\nZatrzymywanie...")
        stop_event.set()
    except Exception as e:
        print(f"Błąd: {e}")
