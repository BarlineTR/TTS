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

def record_mic(duration=5, samplerate=16000, device_id=0):
    """ReSpeaker mikrofonundan belirtilen saniye kadar ses kaydeder"""
    print(f"\n🔴 MİKROFON DİNLENİYOR ({duration} saniye konuşun)...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
    sd.wait()
    print("🟢 DİNLEME TAMAMLANDI!")
    wav.write("user_mic_input.wav", samplerate, audio)
    return "user_mic_input.wav"

def main():
    print("==================================================")
    print("🎙️ ReSpeaker CANLI SESLİ KONUŞMA ASİSTANI (STT + Groq LLM + XTTS v2)")
    print("==================================================")
    
    # GROQ API Anahtarı Kontrolü
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("🔑 Lütfen Groq API Anahtarınızı (gsk_...) girin: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    try:
        groq_client = Groq(api_key=groq_api_key)
        print("✅ Groq API Bağlandı! (Zeka & Whisper Ses Tanıma Aktif)")
    except Exception as e:
        print(f"⚠️ Groq API Bağlantı Hatası: {e}")
        return

    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Mikrofon (ReSpeaker Mic): ID [{input_id}]")
    print(f"🎧 Kulaklık (ReSpeaker Headphone): ID [{output_id}]")
    
    print("\n🧠 XTTS v2 İnsan Benzeri Doğal Ses Modeli Yükleniyor...")
    try:
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=torch.cuda.is_available())
        print("✅ XTTS v2 Doğal Ses Modeli Hazır!")
    except Exception as e:
        print(f"⚠️ XTTS v2 yüklenirken varsayılana dönüldü: {e}")
        tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=torch.cuda.is_available())

    speaker_ref_wav = "respeaker_test.wav"
    output_wav = "full_voice_response.wav"
    target_sample_rate = 16000
    
    messages_history = [
        {
            "role": "system",
            "content": "Sen Jetson Nano üzerinde ReSpeaker ile çalışan akıllı sesli asistansın. Türkçe konuşuyorsun. Yanıtların sesli okunması için kısa, öz ve samimi olsun."
        }
    ]

    print("\n🎙️ TAM SESLİ DÖNGÜ BAŞLADI!")
    print("👉 Enter'a basın -> ReSpeaker mikrofonunuza konuşun -> Yapay zeka sizi dinleyip sesli cevap versin!")
    print("💡 Çıkmak için 'q' yazabilirsiniz.\n")

    while True:
        try:
            cmd = input("\n👉 Konuşmak için [ENTER]'a basın (Çıkış için 'q'): ").strip()
            if cmd.lower() in ['q', 'exit', 'cikis']:
                print("👋 Görüşmek üzere!")
                break

            # 1. ReSpeaker Mikrofonundan Sesi Kaydet
            mic_file = record_mic(duration=5, samplerate=16000, device_id=input_id)

            # 2. Groq Whisper API İle Sesi Metne Çevir (Speech-to-Text)
            print("👂 Sesiniz Anlaşılıyor (Groq Whisper)...")
            with open(mic_file, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(mic_file, file.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )

            user_text = str(transcription).strip()
            print(f"🗣️ Siz (Sesli): \"{user_text}\"")

            if not user_text:
                print("⚠️ Ses anlaşılamadı, lütfen tekrar deneyin.")
                continue

            messages_history.append({"role": "user", "content": user_text})

            # 3. Groq Llama-3.3 LLM Cevap Üretsin
            print("🧠 Düşünüyor (Groq Llama-3.3)...")
            chat_completion = groq_client.chat.completions.create(
                messages=messages_history,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=150
            )

            ai_response = chat_completion.choices[0].message.content.strip()
            messages_history.append({"role": "assistant", "content": ai_response})

            print(f"🤖 Asistan: \"{ai_response}\"")

            # 4. XTTS v2 İle Seslendir (Text-to-Speech)
            print("🗣️ Doğal Türkçe Ses Üretiliyor...")
            if "xtts" in getattr(tts, 'model_name', ''):
                if os.path.exists(speaker_ref_wav):
                    tts.tts_to_file(text=ai_response, speaker_wav=speaker_ref_wav, language="tr", file_path=output_wav)
                else:
                    tts.tts_to_file(text=ai_response, language="tr", file_path=output_wav)
            else:
                tts.tts_to_file(text=ai_response, file_path=output_wav)

            # 5. Sesi ReSpeaker Kulaklığına Çal (Audio Playback)
            rate, data = wav.read(output_wav)
            if rate != target_sample_rate:
                num_samples = int(len(data) * target_sample_rate / rate)
                data = resample(data, num_samples).astype(data.dtype)
                rate = target_sample_rate

            print("🔊 ReSpeaker Kulaklığına Aktarılıyor...")
            sd.play(data, samplerate=rate, device=output_id)
            sd.wait()

        except KeyboardInterrupt:
            print("\n👋 Görüşmek üzere.")
            break
        except Exception as e:
            print(f"⚠️ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
