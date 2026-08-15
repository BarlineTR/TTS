import os
import sys
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from TTS.api import TTS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Coqui Lisansını Otomatik Onayla
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
                
    # Bulunamazsa varsayılan cihazları seç
    if input_id is None:
        input_id = sd.default.device[0]
    if output_id is None:
        output_id = sd.default.device[1]
        
    return input_id, output_id

def main():
    print("==================================================")
    print("🎙️  ReSpeaker İnteraktif Türkçe Sesli Asistan / TTS Demosu")
    print("==================================================")
    
    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Girdi Cihazı (ReSpeaker Mic): ID [{input_id}]")
    print(f"🎧 Çıktı Cihazı (ReSpeaker Headphone): ID [{output_id}]")
    
    print("\n🧠 Türkçe TTS Modeli Yükleniyor (Lütfen Bekleyin)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=torch.cuda.is_available())
    
    print("\n✅ Model Yüklendi! Artık modelle konuşabilir ve yazışabilirsiniz.")
    print("💡 Çıkmak için 'q' veya 'exit' yazabilirsiniz.\n")
    
    output_wav = "interactive_output.wav"
    
    while True:
        try:
            text = input("💬 Ne söylememi istersiniz? (Metin girin): ").strip()
            
            if text.lower() in ['q', 'exit', 'cikis', 'çıkış']:
                print("👋 Görüşmek üzere!")
                break
                
            if not text:
                continue
                
            print("🗣️  Ses Sentezleniyor...")
            tts.tts_to_file(text=text, file_path=output_wav)
            
            # Sesi ReSpeaker Kulaklığından Oynat
            rate, data = wav.read(output_wav)
            print("🔊 ReSpeaker Kulaklığına Ses Aktarılıyor...\n")
            sd.play(data, samplerate=rate, device=output_id)
            sd.wait()
            
        except KeyboardInterrupt:
            print("\n👋 Uygulama sonlandırıldı.")
            break
        except Exception as e:
            print(f"⚠️ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
