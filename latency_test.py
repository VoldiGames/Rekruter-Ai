import asyncio
import websockets
import json
import os
import base64
import pyaudio
import threading
from dotenv import load_dotenv

load_dotenv()

# Konfiguracja audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 512  # małe CHUNK dla płynnego nagrywania

# Event do zatrzymania
stop_event = threading.Event()

# Queue do przesyłania audio między wątkiem a coroutiną
send_queue = asyncio.Queue()


# Wątek nagrywający audio
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("🎤 Nagrywanie... Naciśnij Ctrl+C aby zatrzymać")

    try:
        while not stop_event.is_set():
            data = stream.read(CHUNK, exception_on_overflow=False)
            asyncio.run_coroutine_threadsafe(send_queue.put(data), record_audio.loop)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        stop_event.set()


# Coroutine wysyłająca audio przez WebSocket
async def send_audio(websocket):
    while not stop_event.is_set():
        data = await send_queue.get()
        audio_data = base64.b64encode(data).decode("utf-8")
        message = {"type": "input_audio_buffer.append", "audio": audio_data}
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            print("❌ Błąd wysyłania audio:", e)


# Buforowanie audio przed odtwarzaniem
class AudioBuffer:
    def __init__(self, chunk_limit=48000):  # ok. 2 sekundy przy 24kHz
        self.buffer = bytearray()
        self.chunk_limit = chunk_limit

    def add(self, data):
        self.buffer.extend(data)

    def ready(self):
        return len(self.buffer) >= self.chunk_limit

    def get(self):
        data = bytes(self.buffer)
        self.buffer.clear()
        return data


def play_audio(audio_data):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    try:
        stream.write(audio_data)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def openai_realtime():
    # Wczytanie oferty pracy z pliku
    try:
        with open("oferta.txt", "r", encoding="utf-8") as f:
            oferta_pracy = f.read().strip()
    except FileNotFoundError:
        print("❌ Brak pliku oferta.txt – utwórz go i wklej ofertę pracy.")
        return

    instructions_pl = (
        "Jesteś rekruterem prowadzącym rozmowę kwalifikacyjną. "
        "Rozmowa ma dotyczyć WYŁĄCZNIE poniższej oferty pracy. "
        "Nie wolno Ci odpowiadać na pytania niezwiązane z tą ofertą. "
        "Jeśli użytkownik pyta o coś innego, powiedz krótko: "
        "'Odpowiadam tylko na pytania dotyczące tej oferty pracy.'\n\n"
        f"OFERTA PRACY:\n{oferta_pracy}"
    )

    async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            additional_headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1"
            }
    ) as websocket:

        # Inicjalizacja sesji
        await websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": instructions_pl
            }
        }))

        # Start wątku nagrywania
        record_audio.loop = asyncio.get_event_loop()
        record_thread = threading.Thread(target=record_audio)
        record_thread.daemon = True
        record_thread.start()

        # Coroutine do wysyłania audio
        asyncio.create_task(send_audio(websocket))

        # Poczekaj 1 sekundę przed pierwszą odpowiedzią
        await asyncio.sleep(1)
        print("✅ Rozmowa rozpoczęta. Mów do mikrofonu...")

        audio_buffer = AudioBuffer(chunk_limit=RATE * 1)  # 1 sekunda audio

        try:
            async for message in websocket:
                data = json.loads(message)

                if data['type'] == 'response.audio.delta':
                    audio_bytes = base64.b64decode(data['delta'])
                    audio_buffer.add(audio_bytes)

                    if audio_buffer.ready():
                        play_audio(audio_buffer.get())

                elif data['type'] == 'response.content_part.done':
                    if 'transcript' in data['part']:
                        print(f"Asystent: {data['part']['transcript']}")

                elif data['type'] == 'error':
                    print(f"❌ Błąd: {data['error']}")

                elif data['type'] == 'session.updated':
                    print("ℹ️ Sesja zainicjalizowana")

                elif data['type'] == 'response.created':
                    print("🔄 Otrzymywanie odpowiedzi...")

        except Exception as e:
            print(f"❌ Błąd połączenia: {e}")
        finally:
            stop_event.set()
            try:
                record_thread.join(timeout=1.0)
            except:
                pass


if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: Ustaw zmienną środowiskową OPENAI_API_KEY")
        exit(1)

    asyncio.run(openai_realtime())
