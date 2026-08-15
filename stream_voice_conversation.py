import os
import sys
import time
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import resample
from groq import Groq
from TTS.api import TTS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["COQUI_TOS_AGREED"] = "1"

def find_respeaker_devices():
    devices = sd.query_devices()
    input_id = None
    output_id = None
    
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if ("respeaker" in name or "uac1" in name or "seeed" in name):
            if dev['max_input_channels'] > 0 and input_id is None:
                input_id = i
            if dev['max_output_channels'] > 0 and output_id is None:
                output_id = i
                
    if input_id is None:
        input_id = sd.default.device[0]
    if output_id is None:
        output_id = sd.default.device[1]
        
    return input_id, output_id

def record_mic(duration=4, samplerate=16000, device_id=0):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
    sd.wait()
    wav.write("user_mic_input.wav", samplerate, audio)
    return "user_mic_input.wav"

def clean_text_for_tts(text):
    replacements = {
        'Ç': 'c', 'ç': 'c', 'Ğ': 'g', 'ğ': 'g', 'I': 'i', 'ı': 'i', 'İ': 'i',
        'Ö': 'o', 'ö': 'o', 'Ş': 's', 'ş': 's', 'Ü': 'u', 'ü': 'u',
        'Q': 'k', 'q': 'k', 'W': 'v', 'w': 'v', 'X': 'ks', 'x': 'ks'
    }
    text = text.lower()
    for old, new in replacements.items():
        text = text.replace(old.lower(), new)
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?"
    return "".join([c if c in allowed else ' ' for c in text])

def main():
    print("==================================================")
    print("⚡ Real-Time Streaming Voice Assistant (1-Second Latency)")
    print("==================================================")
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("🔑 Groq API Key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    input_id, output_id = find_respeaker_devices()
    
    print("\n⚡ Yükleniyor...")
    tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=False)
    print("✅ Hazır!")

    target_sample_rate = 16000
    messages_history = [
        {
            "role": "system",
            "content": "Sen zeki, samimi ve bilgili bir sesli asistansın. Türkçe konuş. Sorulara eksiksiz, doğal ve akıcı cevaplar ver."
        }
    ]

    while True:
        try:
            cmd = input("\n👉 Konuşmak için [ENTER] (Çıkış: 'q'): ").strip()
            if cmd.lower() in ['q', 'exit']:
                break

            t0 = time.time()
            mic_file = record_mic(duration=4, samplerate=16000, device_id=input_id)

            # 1. Groq Whisper - Hızlı Ses Tanıma
            with open(mic_file, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(mic_file, file.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )

            user_text = str(transcription).strip()
            if not user_text:
                continue

            print(f"🗣️ Siz: \"{user_text}\"")
            messages_history.append({"role": "user", "content": user_text})

            # 2. STREAMING LLM (Cümle Cümle Akış & Anında Ses Sentezi)
            print("⚡ Akış Başladı...")
            stream = groq_client.chat.completions.create(
                messages=messages_history,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                stream=True
            )

            full_response = ""
            sentence_buffer = ""

            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                sentence_buffer += content

                # Cümle bittiğinde (nokta, soru işareti veya ünlem görünce) ANINDA SESE ÇEVİR VE OKU
                if any(p in sentence_buffer for p in ['.', '?', '!', '\n']):
                    clean_sentence = clean_text_for_tts(sentence_buffer.strip())
                    if clean_sentence.strip():
                        print(f"🤖 (Canlı): {sentence_buffer.strip()}")
                        chunk_wav = "stream_chunk.wav"
                        tts.tts_to_file(text=clean_sentence, file_path=chunk_wav)
                        
                        # Ses Dosyasını ReSpeaker'a Oynat
                        rate, data = wav.read(chunk_wav)
                        if rate != target_sample_rate:
                            num_samples = int(len(data) * target_sample_rate / rate)
                            data = resample(data, num_samples).astype(data.dtype)
                            rate = target_sample_rate

                        sd.play(data, samplerate=rate, device=output_id)
                        sd.wait()

                    sentence_buffer = ""

            # Kalan son cümleyi de seslendir
            if sentence_buffer.strip():
                clean_sentence = clean_text_for_tts(sentence_buffer.strip())
                if clean_sentence.strip():
                    chunk_wav = "stream_chunk.wav"
                    tts.tts_to_file(text=clean_sentence, file_path=chunk_wav)
                    rate, data = wav.read(chunk_wav)
                    if rate != target_sample_rate:
                        num_samples = int(len(data) * target_sample_rate / rate)
                        data = resample(data, num_samples).astype(data.dtype)
                        rate = target_sample_rate
                    sd.play(data, samplerate=rate, device=output_id)
                    sd.wait()

            messages_history.append({"role": "assistant", "content": full_response})
            print(f"⏱️ Toplam Yanıta Başlama & İşleme Süresi: {round(time.time() - t0, 2)}s")

        except Exception as e:
            print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    main()
