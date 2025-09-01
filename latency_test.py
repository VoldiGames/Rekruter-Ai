import asyncio
import websockets
import json
import os
import base64
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from dotenv import load_dotenv
import requests
import time

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Brak OPENAI_API_KEY w pliku .env")

MODEL_REALTIME = "gpt-4o-realtime-preview"
URL_REALTIME = f"wss://api.openai.com/v1/realtime?model={MODEL_REALTIME}"
MODEL_TTS = "gpt-4o-mini-tts"
OUTPUT_FILE = r"C:\Users\tomek\PycharmProjects\RekruterAi\response.mp3"

DURATION = 5
ORIGINAL_SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
MIC_DEVICE = 15
sd.default.device = MIC_DEVICE


def generate_tts(text, output_path):
    """
    Generuje TTS — staramy się wymusić język polski poprzez nagłówki i pola payloadu.
    Jeśli endpoint nie obsługuje pola 'language', kluczowe jest aby 'text' był po polsku.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept-Language": "pl-PL"
    }
    payload = {"model": MODEL_TTS, "voice": "alloy", "language": "pl-PL", "input": text}
    response = requests.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Audio TTS zapisane w {output_path}")
    else:
        print("Błąd TTS:", response.status_code, response.text)


async def audio_to_audio():
    print("Łączenie z WebSocket...")
    async with websockets.connect(
            URL_REALTIME,
            additional_headers={
                "Authorization": f"Bearer {API_KEY}",
                "OpenAI-Beta": "realtime=v1",
                "Accept-Language": "pl-PL"
            }
    ) as ws:
        print("Tworzenie odpowiedzi...")

        # Wyraźne instrukcje systemowe — wymuszamy PL
        instructions_pl = (
            "Jesteś asystentem. ODPOWIADAJ WYŁĄCZNIE PO POLSKU. "
            "Nawet jeśli użytkownik użyje innego języka, zawsze odpowiadaj po polsku. "
            "Nie dodawaj wstępu ani komentarzy. Odpowiadaj krótko i konkretnie oraz podawaj tylko to, o co prosi użytkownik."
        )

        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "audio": {"voice": "nova", "output_format": "mp3"},
                "instructions": instructions_pl
            }
        }))

        print("Nagrywanie audio...")
        recording = sd.rec(int(DURATION * ORIGINAL_SAMPLE_RATE), samplerate=ORIGINAL_SAMPLE_RATE, channels=1,
                           dtype='int16')
        sd.wait()
        print("Nagrywanie zakończone")

        num_samples = int(len(recording) * TARGET_SAMPLE_RATE / ORIGINAL_SAMPLE_RATE)
        resampled = resample(recording, num_samples).astype(np.int16)
        audio_bytes = resampled.tobytes()

        print("Wysyłanie audio strumieniowo...")
        chunk_size = TARGET_SAMPLE_RATE // 2
        for i in range(0, len(audio_bytes), chunk_size * 2):
            chunk = audio_bytes[i:i + chunk_size * 2]
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk_b64}))

        start_time = time.perf_counter()
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("Audio wysłane")

        full_text_response = ""
        first_token_time = None

        async for msg in ws:
            data = json.loads(msg)
            event_type = data.get("type")
            print("Otrzymano event:", event_type)

            if event_type in ["response.text.delta", "response.audio_transcript.delta"]:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    latency_ms = (first_token_time - start_time) * 1000
                    print(f"\n⏱️ Opóźnienie pierwszego tokena: {latency_ms:.2f} ms")
                full_text_response += data.get("delta", "")
                print(data.get("delta", ""), end="")

            elif event_type == "response.done":
                print("\nTranskrypcja końcowa:", full_text_response)
                break

        if full_text_response:
            # Dodatkowo: można tu wrzucić detekcję języka (langdetect) i automatyczne tłumaczenie do PL,
            # gdyby model mimo wszystko odpowiedział w innym języku.
            generate_tts(full_text_response, OUTPUT_FILE)


if __name__ == "__main__":
    asyncio.run(audio_to_audio())