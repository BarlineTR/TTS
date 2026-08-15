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

def main():
    print("==================================================")
    print("🚀 Jetson Nano Ultra-Hızlı Türkçe Sesli Asistan (Glow-TTS + Groq)")
    print("==================================================")
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("🔑 Lütfen Groq API Anahtarınızı (gsk_...) girin: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    try:
        groq_client = Groq(api_key=groq_api_key)
        print("✅ Groq API Bağlandı!")
    except Exception as e:
        print(f"⚠️ Groq API Bağlantı Hatası: {e}")
        return

    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Mikrofon ID: [{input_id}] | 🎧 Kulaklık ID: [{output_id}]")
    
    # JETSON NANO İÇİN ULTRA HIZLI TÜRKÇE TTS MODELİ (Glow-TTS)
    print("\n⚡ Ultra-Hızlı Türkçe TTS Modeli (Glow-TTS) Yükleniyor...")
    tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=False)
    print("✅ Hızlı Ses Modeli Hazır!")

    output_wav = "fast_voice_response.wav"
    target_sample_rate = 16000
    
    messages_history = [
        {
            "role": "system",
            "content": "Sen Jetson Nano üzerinde çalışan ultra hızlı sesli asistansın. Türkçe cevap ver. Cevapların çok kısa (maksimum 1-2 kısa cümle) olsun ki hızlı seslendirilsin."
        }
    ]

    print("\n🎙️ IŞIK HIZINDA SESLİ SOHBET BAŞLADI!")
    print("👉 Enter'a basın -> Konuşun -> Anında Yanıt Dinleyin!")
    print("💡 Çıkış için 'q'\n")

    while True:
        try:
            cmd = input("\n👉 Konuşmak için [ENTER]'a basın (Çıkış: 'q'): ").strip()
            if cmd.lower() in ['q', 'exit', 'cikis']:
                print("👋 Görüşmek üzere!")
                break

            # 1. 4 Saniyelik Ses Kaydı
            mic_file = record_mic(duration=4, samplerate=16000, device_id=input_id)

            # 2. Groq Whisper (Saliseler İçinde Metne Çevirir)
            print("👂 Ses Metne Çevriliyor...")
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
                print("⚠️ Ses algılanamadı.")
                continue

            messages_history.append({"role": "user", "content": user_text})

            # 3. Groq Llama-3.3 (0.2 saniyede cevap üretir)
            print("⚡ Groq Yanıt Üretiyor...")
            chat_completion = groq_client.chat.completions.create(
                messages=messages_history,
                model="llama-3.3-70b-versatile",
                temperature=0.6,
                max_tokens=80  # Kısa yanıtlar ile süreyi minimuma indirme
            )

            ai_response = chat_completion.choices[0].message.content.strip()
            messages_history.append({"role": "assistant", "content": ai_response})

            print(f"🤖 Asistan: \"{ai_response}\"")

            # 4. Hızlı Türkçe Ses Sentezi (0.5 saniye)
            print("🗣️ Hızlı Ses Üretiliyor...")
            tts.tts_to_file(text=ai_response, file_path=output_wav)

            # 5. Sesi Oynat
            rate, data = wav.read(output_wav)
            if rate != target_sample_rate:
                num_samples = int(len(data) * target_sample_rate / rate)
                data = resample(data, num_samples).astype(data.dtype)
                rate = target_sample_rate

            print("🔊 Çalınıyor...")
            sd.play(data, samplerate=rate, device=output_id)
            sd.wait()

        except KeyboardInterrupt:
            print("\n👋 Görüşmek üzere.")
            break
        except Exception as e:
            print(f"⚠️ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
