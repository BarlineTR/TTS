import os
import sys
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
    print(f"\n🔴 MİKROFON DİNLENİYOR ({duration} saniye konuşun)...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
    sd.wait()
    print("🟢 DİNLEME TAMAMLANDI!")
    wav.write("user_mic_input.wav", samplerate, audio)
    return "user_mic_input.wav"

def clean_text_for_tts(text):
    """Büyük harfleri ve Türkçe harfleri Glow-TTS vokabülerini bozmayacak şekilde temizler"""
    replacements = {
        'Ç': 'c', 'ç': 'c',
        'Ğ': 'g', 'ğ': 'g',
        'I': 'i', 'ı': 'i',
        'İ': 'i',
        'Ö': 'o', 'ö': 'o',
        'Ş': 's', 'ş': 's',
        'Ü': 'u', 'ü': 'u',
        'Q': 'k', 'q': 'k',
        'W': 'v', 'w': 'v',
        'X': 'ks', 'x': 'ks'
    }
    
    text = text.lower()
    for old, new in replacements.items():
        text = text.replace(old.lower(), new)
        
    # Sadece alfabedeki harfleri ve temel noktalama işaretlerini tut
    clean_chars = []
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?"
    for char in text:
        if char in allowed:
            clean_chars.append(char)
        else:
            clean_chars.append(' ')
            
    return "".join(clean_chars)

def main():
    print("==================================================")
    print("🚀 Optimize Edilmiş Jetson Nano Sesli Asistan v2")
    print("==================================================")
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("🔑 Lütfen Groq API Anahtarınızı girin: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Mikrofon ID: [{input_id}] | 🎧 Kulaklık ID: [{output_id}]")
    
    print("\n⚡ Türkçe Glow-TTS Modeli Yükleniyor...")
    tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=False)
    print("✅ Model Hazır!")

    output_wav = "optimized_voice_response.wav"
    target_sample_rate = 16000
    
    messages_history = [
        {
            "role": "system",
            "content": "Sen Jetson üzerinde çalışan neşeli bir asistansın. Türkçe konuşuyorsun. Sadece Türkçe kelimeler kullan. İngilizce kelimeler kesinlikle kullanma. Cevabın çok kısa olsun (en fazla 10-12 kelime)."
        }
    ]

    print("\n🎙️ HIZLI VE KUSURSUZ SOHBET BAŞLADI!")

    while True:
        try:
            cmd = input("\n👉 Konuşmak için [ENTER]'a basın (Çıkış: 'q'): ").strip()
            if cmd.lower() in ['q', 'exit', 'cikis']:
                break

            mic_file = record_mic(duration=4, samplerate=16000, device_id=input_id)

            print("👂 Ses Anlaşılıyor...")
            with open(mic_file, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(mic_file, file.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )

            user_text = str(transcription).strip()
            print(f"🗣️ Siz: \"{user_text}\"")

            if not user_text:
                continue

            messages_history.append({"role": "user", "content": user_text})

            print("⚡ Groq Yanıt Üretiyor...")
            chat_completion = groq_client.chat.completions.create(
                messages=messages_history,
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_tokens=40  # Süreyi düşürmek için yanıt boyutunu sınırlandırma
            )

            raw_ai_response = chat_completion.choices[0].message.content.strip()
            messages_history.append({"role": "assistant", "content": raw_ai_response})
            print(f"🤖 Asistan: \"{raw_ai_response}\"")

            # 1. Kelime Yutmayı Önleyen Metin Temizleyici (Text Normalization)
            cleaned_response = clean_text_for_tts(raw_ai_response)
            
            # 2. Hızlı Ses Sentezi
            print("🗣️ Hızlı Ses Üretiliyor...")
            tts.tts_to_file(text=cleaned_response, file_path=output_wav)

            # 3. Ses Oynatma
            rate, data = wav.read(output_wav)
            if rate != target_sample_rate:
                num_samples = int(len(data) * target_sample_rate / rate)
                data = resample(data, num_samples).astype(data.dtype)
                rate = target_sample_rate

            print("🔊 Çalınıyor...")
            sd.play(data, samplerate=rate, device=output_id)
            sd.wait()

        except Exception as e:
            print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    main()
