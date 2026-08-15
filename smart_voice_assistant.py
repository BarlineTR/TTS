import os
import sys
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
from scipy.signal import resample
from google.genai import types
from google import genai
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
    print("🧠 ReSpeaker + Akıllı LLM Yapay Zeka Asistanı")
    print("==================================================")
    
    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Girdi Cihazı (ReSpeaker Mic): ID [{input_id}]")
    print(f"🎧 Çıktı Cihazı (ReSpeaker Headphone): ID [{output_id}]")
    
    # XTTS v2 Doğal ve Akıcı Ses Modeli Yükleme
    print("\n🎭 Üst Seviye Doğal Ses Modeli (XTTS v2) Yükleniyor...")
    try:
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=torch.cuda.is_available())
        print("✅ XTTS v2 Doğal Ses Modeli Başarıyla Yüklendi!")
    except Exception as e:
        print(f"⚠️ XTTS v2 yüklenirken varsayılana dönüldü: {e}")
        tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=torch.cuda.is_available())

    # Referans Ses Dosyası (Ses Klonlama İçin ReSpeaker Kaydınız)
    speaker_ref_wav = "respeaker_test.wav"
    if not os.path.exists(speaker_ref_wav):
        print("💡 Not: Kendi doğal ses tonunuzla konuşması için önce 'record_respeaker.py' ile ses kaydedebilirsiniz.")
    
    output_wav = "assistant_response.wav"
    target_sample_rate = 16000
    
    print("\n💬 Asistan Hazır! Sorunuzu yazın, yapay zeka düşünüp doğal sesiyle cevap versin.")
    print("💡 Çıkmak için 'q' yazabilirsiniz.\n")

    # Basit Akıllı Sohbet Döngüsü
    while True:
        try:
            user_input = input("👤 Siz: ").strip()
            
            if user_input.lower() in ['q', 'exit', 'cikis', 'çıkış']:
                print("👋 Görüşmek üzere!")
                break
                
            if not user_input:
                continue

            # Basit Cevaplama Mantığı (Gerektiğinde LLM API veya Kural Tabanlı Mantık)
            print("🧠 Düşünüyor...")
            
            # Örnek Akıllı Mantık (Daha sonra Ollama / Gemini / Local LLM bağlanabilir)
            if "naber" in user_input.lower() or "nasılsın" in user_input.lower():
                ai_response = "Harikayım, teşekkür ederim! ReSpeaker mikrofonunuz ve doğal ses sisteminizle çalışıyorum. Siz nasılsınız?"
            elif "kimsin" in user_input.lower() or "adın ne" in user_input.lower():
                ai_response = "Ben Jetson Nano üzerinde çalışan akıllı sesli asistanınızım."
            else:
                ai_response = f"Söylediğiniz '{user_input}' konusunu anladım. Size nasıl yardımcı olabilirim?"

            print(f"🤖 Asistan: {ai_response}")
            print("🗣️  Doğal Ses Sentezleniyor...")

            # XTTS v2 ile Seslendirme (Eğer mevcutsa klonlanmış doğal sesle)
            if hasattr(tts, 'speaker_manager') or "xtts" in getattr(tts, 'model_name', ''):
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

            # Sesi Oynat ve ReSpeaker'a ilet
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
