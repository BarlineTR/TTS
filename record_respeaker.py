import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def record_audio(filename="respeaker_test.wav", duration=5, samplerate=16000):
    devices = sd.query_devices()
    device_id = None
    
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if ("respeaker" in name or "seeed" in name) and dev['max_input_channels'] > 0:
            device_id = i
            break
            
    if device_id is None:
        device_id = sd.default.device[0]
        print(f"[INFO] Cihaz: ID [{device_id}] - ({devices[device_id]['name']})")
    else:
        print(f"[SUCCESS] ReSpeaker Mikrofonu Secildi: ID [{device_id}] - ({devices[device_id]['name']})")

    print(f"\n🔴 KAYIT BAŞLADI ({duration} saniye boyunca ReSpeaker mikrofonuna konuşun)...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
    sd.wait()
    print("🟢 KAYIT TAMAMLANDI!")

    wav.write(filename, samplerate, audio)
    print(f"💾 Ses dosyası kaydedildi: {os.path.abspath(filename)}")
    return filename

if __name__ == "__main__":
    record_audio()
