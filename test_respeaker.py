import sounddevice as sd
import sys

# Windows konsol unicode çıktılarını düzeltmek için:
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def detect_respeaker():
    print("=== Bagli Ses Cihazlari ===")
    devices = sd.query_devices()
    respeaker_input = None
    
    for i, dev in enumerate(devices):
        max_in = dev['max_input_channels']
        max_out = dev['max_output_channels']
        name = dev['name']
        print(f"[{i}] {name} (Girdi Kanali: {max_in}, Cikti Kanali: {max_out})")
        
        # ReSpeaker veya USB Audio cihaz arama
        if ("respeaker" in name.lower() or "seeed" in name.lower() or "usb audio" in name.lower() or "mic" in name.lower()) and max_in > 0:
            if respeaker_input is None or "respeaker" in name.lower():
                respeaker_input = i
                
    print("\n----------------------------------------")
    if respeaker_input is not None:
        print(f"[SUCCESS] ReSpeaker / Mikrofon Cihazi Tespit Edildi: ID [{respeaker_input}] - {devices[respeaker_input]['name']}")
    else:
        print("[WARNING] Ozel adıyla ReSpeaker bulunamadi, ancak varsayilan mikrofon kullanilabilir.")
        default_in = sd.default.device[0]
        print(f"Varsayilan Girdi Cihazi ID: [{default_in}] - {devices[default_in]['name']}")
    print("----------------------------------------")

if __name__ == "__main__":
    detect_respeaker()
