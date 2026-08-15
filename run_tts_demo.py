import os
import sys
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
from TTS.api import TTS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def run_tts_demo():
    print("=== TTS / Ses Sentezleme & Klonlama Testi ===")
    
    # 1. Türkçe TTS Testi (Standart Ön Eğitimli Model)
    text_tr = "Merhaba! ReSpeaker mikrofonunuz ve Coqui TTS ses modeliniz başarıyla entegre edildi. Test başarılı."
    output_tr_wav = "tts_output_tr.wav"
    
    print("\n1️⃣  Türkçe TTS Modeli Yükleniyor (tts_models/tr/common-voice/glow-tts)...")
    try:
        tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=torch.cuda.is_available())
        print("🗣️  Türkçe Ses Sentezleniyor...")
        tts.tts_to_file(text=text_tr, file_path=output_tr_wav)
        print(f"✅ Türkçe Ses Oluşturuldu: {os.path.abspath(output_tr_wav)}")
        
        # Sesi Oynat
        rate, data = wav.read(output_tr_wav)
        print("🔊 Ses Oynatılıyor...")
        sd.play(data, samplerate=rate)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Türkçe TTS Test Hatası: {e}")

    # 2. ReSpeaker Ses Klonlama Testi (XTTS v2 / Multi-Speaker / Voice Clone)
    ref_wav = "respeaker_test.wav"
    if os.path.exists(ref_wav):
        print("\n2️⃣  ReSpeaker Mikrofon Kaydınız İle Ses Klonlama Testi Yapılıyor...")
        text_clone = "Bu ses, ReSpeaker mikrofonunuzdan alınan ses örneği kullanılarak klonlanmıştır."
        output_clone_wav = "tts_output_clone.wav"
        
        try:
            print("🧠 Ses Klonlama Modeli (XTTS v2) Yükleniyor...")
            tts_clone = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=torch.cuda.is_available())
            print("🎭 Sesiniz Klonlanıyor ve Konuşturuluyor...")
            tts_clone.tts_to_file(
                text=text_clone,
                speaker_wav=ref_wav,
                language="tr",
                file_path=output_clone_wav
            )
            print(f"✅ Klonlanmış Ses Dosyası Oluşturuldu: {os.path.abspath(output_clone_wav)}")
            
            # Sesi Oynat
            rate, data = wav.read(output_clone_wav)
            print("🔊 Klon Ses Oynatılıyor...")
            sd.play(data, samplerate=rate)
            sd.wait()
        except Exception as e:
            print(f"⚠️ Ses Klonlama Test Hatası: {e}")
    else:
        print("\n💡 Ses Klonlama Testi İçin Önceden 'record_respeaker.py' Çalıştırıp Ses Kaydetmelisiniz.")

if __name__ == "__main__":
    run_tts_demo()
