import os
import sys
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
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

def main():
    print("==================================================")
    print("⚡ Groq LLM + ReSpeaker + XTTS v2 Doğal Sesli Asistan")
    print("==================================================")
    
    # GROQ API Anahtarı Kontrolü
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("🔑 Lütfen Groq API Anahtarınızı (gsk_...) girin: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    try:
        groq_client = Groq(api_key=groq_api_key)
        print("✅ Groq API İle Bağlantı Kuruldu! (Model: llama-3.3-70b-versatile / llama3-8b-8192)")
    except Exception as e:
        print(f"⚠️ Groq API Bağlantı Hatası: {e}")
        return

    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Girdi Cihazı (ReSpeaker Mic): ID [{input_id}]")
    print(f"🎧 Çıktı Cihazı (ReSpeaker Headphone): ID [{output_id}]")
    
    print("\n🧠 XTTS v2 İnsan Benzeri Doğal Ses Modeli Yükleniyor...")
    try:
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=torch.cuda.is_available())
        print("✅ XTTS v2 Doğal Ses Modeli Hazır!")
    except Exception as e:
        print(f"⚠️ XTTS v2 yüklenirken varsayılana dönüldü: {e}")
        tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=torch.cuda.is_available())

    speaker_ref_wav = "respeaker_test.wav"
    output_wav = "groq_assistant_response.wav"
    target_sample_rate = 16000
    
    messages_history = [
        {
            "role": "system",
            "content": "Sen Jetson Nano üzerinde çalışan, ReSpeaker mikrofonu ve Türkçe ses sentezleyici kullanan son derece zeki, samimi ve Türkçe konuşan sesli bir yapay zeka asistanısın. Yanıtlarını sesli okunmaya uygun, çok uzun olmayan, akıcı cümlelerle ver."
        }
    ]

    print("\n🎙️ Asistanınız Dinlemede! Sorunuzu sorun, Groq anında düşünsün ve konuşsun.")
    print("💡 Çıkmak için 'q' yazabilirsiniz.\n")

    while True:
        try:
            user_input = input("👤 Siz: ").strip()
            
            if user_input.lower() in ['q', 'exit', 'cikis', 'çıkış']:
                print("👋 Görüşmek üzere!")
                break
                
            if not user_input:
                continue

            messages_history.append({"role": "user", "content": user_input})

            print("⚡ Groq Yapay Zekası Düşünüyor...")
            
            chat_completion = groq_client.chat.completions.create(
                messages=messages_history,
                model="llama-3.3-70b-versatile",  # Çok hızlı ve zeki Llama 3.3 modeli
                temperature=0.7,
                max_tokens=150
            )

            ai_response = chat_completion.choices[0].message.content.strip()
            messages_history.append({"role": "assistant", "content": ai_response})

            print(f"🤖 Groq Asistan: {ai_response}")
            print("🗣️  XTTS v2 Doğal Türkçe Ses Sentezleniyor...")

            if "xtts" in getattr(tts, 'model_name', ''):
                if os.path.exists(speaker_ref_wav):
                    tts.tts_to_file(
                        text=ai_response,
                        speaker_wav=speaker_ref_wav,
                        language="tr",
                        file_path=output_wav
                    )
                else:
                    tts.tts_to_file(
                        text=ai_response,
                        language="tr",
                        file_path=output_wav
                    )
            else:
                tts.tts_to_file(text=ai_response, file_path=output_wav)

            # Sesi Oynat ve ReSpeaker'a İlet
            rate, data = wav.read(output_wav)
            if rate != target_sample_rate:
                num_samples = int(len(data) * target_sample_rate / rate)
                data = resample(data, num_samples).astype(data.dtype)
                rate = target_sample_rate

            print("🔊 ReSpeaker Kulaklığına Aktarılıyor...\n")
            sd.play(data, samplerate=rate, device=output_id)
            sd.wait()

        except KeyboardInterrupt:
            print("\n👋 Görüşmek üzere.")
            break
        except Exception as e:
            print(f"⚠️ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
