import asyncio
import websockets
import json
import os
import base64
import numpy as np
import pyaudio
import threading
import queue
from dotenv import load_dotenv

load_dotenv()
# Konfiguracja audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

# Kolejki do komunikacji między wątkami
audio_queue = queue.Queue()
stop_event = threading.Event()


# Funkcja do nagrywania audio
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
            data = stream.read(CHUNK)
            audio_data = base64.b64encode(data).decode('utf-8')

            # Wysyłanie audio przez WebSocket
            if hasattr(record_audio, 'websocket'):
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_data
                }
                asyncio.run_coroutine_threadsafe(
                    record_audio.websocket.send(json.dumps(message)),
                    record_audio.loop
                )

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Błąd nagrywania: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        stop_event.set()


# Funkcja do odtwarzania audio
def play_audio(audio_data):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)

    try:
        stream.write(audio_data)
    except Exception as e:
        print(f"Błąd odtwarzania: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# Główna funkcja do komunikacji z OpenAI
async def openai_realtime():
    instructions_pl = (
        "Jesteś asystentem. ODPOWIADAJ WYŁĄCZNIE PO POLSKU. "
        "Nigdy nie zmieniaj języka — nawet jeśli użytkownik użyje innego języka, "
        "zawsze odpowiedz po polsku. Nie dodawaj wstępu ani komentarzy. "
        "Odpowiadaj krótko i konkretnie oraz podawaj tylko to, o co prosi użytkownik."
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

        # Przekazanie websocket do wątku nagrywania
        record_audio.websocket = websocket
        record_audio.loop = asyncio.get_event_loop()

        # Rozpocznij wątek nagrywania
        record_thread = threading.Thread(target=record_audio)
        record_thread.daemon = True
        record_thread.start()

        print("Rozmowa rozpoczęta. Mów do mikrofonu...")

        try:
            async for message in websocket:
                data = json.loads(message)

                if data['type'] == 'response.audio.delta':
                    # Odtwarzanie otrzymanego audio
                    audio_data = base64.b64decode(data['delta'])
                    play_audio(audio_data)

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
            try:
                record_thread.join(timeout=1.0)
            except:
                pass


if __name__ == "__main__":
    # Sprawdzenie czy API key jest ustawione
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: Ustaw zmienną środowiskową OPENAI_API_KEY")
        print("Przykład: export OPENAI_API_KEY='twój-klucz-api'")
        exit(1)

    asyncio.run(openai_realtime())
