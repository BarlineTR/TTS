import os
import sys
import asyncio
import tempfile
import threading
import subprocess
import time
import queue
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import resample
from groq import Groq
import edge_tts

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["COQUI_TOS_AGREED"] = "1"

# ─── Ayarlar ──────────────────────────────────────────────────────────────────
EDGE_TTS_VOICE    = "tr-TR-AhmetNeural"   # tr-TR-EmelNeural (kadın ses)
SAMPLE_RATE       = 16000
WAKE_WORDS        = [
    "hey groq", "hey grok", "hey grup", "hey krog", "hey crock",
    "a groq", "a grok", "groq", "grok", "hi groq"
]
VAD_SILENCE_SECS  = 1.0    # Bu kadar sessizlik → konuşma bitti say
VAD_THRESHOLD     = 500    # Ses amplitude eşiği (RMS)
MAX_RECORD_SECS   = 8      # Maksimum kayıt süresi
CHUNK_SECS        = 0.1    # Streaming blok boyutu
# ──────────────────────────────────────────────────────────────────────────────

def find_respeaker():
    devices = sd.query_devices()
    in_id, out_id = None, None
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if any(k in name for k in ["respeaker", "uac1", "seeed"]):
            if dev['max_input_channels'] > 0 and in_id is None:
                in_id = i
            if dev['max_output_channels'] > 0 and out_id is None:
                out_id = i
    # 0 falsy olduğu için None karşılaştırması zorunlu
    in_id  = in_id  if in_id  is not None else sd.default.device[0]
    out_id = out_id if out_id is not None else sd.default.device[1]
    return in_id, out_id


def record_vad(device_id, max_secs=MAX_RECORD_SECS, debug=False) -> np.ndarray:
    """
    VAD (Ses Aktivite Algılama) tabanlı dinamik kayıt.
    Konuşma başlayana kadar bekler, konuşma bitince durur.
    """
    chunk = int(SAMPLE_RATE * CHUNK_SECS)
    frames = []
    silent_chunks = 0
    speaking = False
    max_chunks = int(max_secs / CHUNK_SECS)
    peak_rms = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype='int16', device=device_id, blocksize=chunk) as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk)
            rms = int(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            peak_rms = max(peak_rms, rms)

            if debug:
                bar = '█' * min(int(rms / 100), 20)
                sys.stdout.write(f"  🎙️  RMS: {rms:5d} [{bar:<20}] eşik={VAD_THRESHOLD}\r")
                sys.stdout.flush()

            if rms > VAD_THRESHOLD:
                speaking = True
                silent_chunks = 0
                frames.append(data)
            elif speaking:
                frames.append(data)
                silent_chunks += 1
                if silent_chunks > (VAD_SILENCE_SECS / CHUNK_SECS):
                    break

    if debug:
        print(f"  📊 Peak RMS: {peak_rms}  |  Eşik: {VAD_THRESHOLD}")

    if not frames:
        return np.zeros((0, 1), dtype='int16')
    return np.concatenate(frames, axis=0)


def audio_to_wav_file(audio_np: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio_np)
    return tmp.name


def edge_speak(text: str, out_id: int):
    """Edge-TTS ile sentezle → ffmpeg WAV → sounddevice ile çal"""
    if not text.strip():
        return
    mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    wav_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    mp3.close(); wav_f.close()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def _synth():
        await edge_tts.Communicate(text, EDGE_TTS_VOICE).save(mp3.name)
    loop.run_until_complete(_synth())
    loop.close()

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3.name,
         "-ar", str(SAMPLE_RATE), "-ac", "1", wav_f.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        rate, data = wav.read(wav_f.name)
        sd.play(data, samplerate=rate, device=out_id)
        sd.wait()
    finally:
        os.unlink(mp3.name)
        os.unlink(wav_f.name)


# ─── LLM Akışlı Yanıt + Eşzamanlı TTS ───────────────────────────────────────
def stream_and_speak(groq_client, messages, out_id):
    stream = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        stream=True
    )

    full_response = ""
    buffer = ""
    play_thread = None

    def play(sentence):
        edge_speak(sentence, out_id)

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_response += token
        buffer += token

        if any(p in buffer for p in ['.', '?', '!', '\n']):
            sentence = buffer.strip()
            buffer = ""
            if sentence:
                print(f"  🤖 {sentence}")
                if play_thread and play_thread.is_alive():
                    play_thread.join()
                play_thread = threading.Thread(target=play, args=(sentence,), daemon=True)
                play_thread.start()

    if buffer.strip():
        if play_thread and play_thread.is_alive():
            play_thread.join()
        edge_speak(buffer.strip(), out_id)

    if play_thread and play_thread.is_alive():
        play_thread.join()

    return full_response
# ──────────────────────────────────────────────────────────────────────────────


def main():
    # Banner
    print("\033[94m" + """
  ██╗  ██╗███████╗██╗   ██╗     ██████╗ ██████╗  ██████╗  ██████╗
  ██║  ██║██╔════╝╚██╗ ██╔╝    ██╔════╝ ██╔══██╗██╔═══██╗██╔═══██╗
  ███████║█████╗   ╚████╔╝     ██║  ███╗██████╔╝██║   ██║██║   ██║
  ██╔══██║██╔══╝    ╚██╔╝      ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║
  ██║  ██║███████╗   ██║       ╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝
  ╚═╝  ╚═╝╚══════╝   ╚═╝        ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝
  Jetson Sesli Asistan │ Edge-TTS + Groq Whisper + LLaMA 3.3
    """ + "\033[0m")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("  🔑 Groq API Key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    in_id, out_id = find_respeaker()

    print(f"  🎤 Mikrofon  : [{in_id}]  |  🎧 Kulaklık : [{out_id}]")
    print(f"  🗣️  TTS Sesi  : {EDGE_TTS_VOICE}")
    print(f"  ⚡ LLM       : llama-3.3-70b-versatile (Groq)\n")

    # VAD Eşiği Kalibrasyonu
    print("  🔊 Ses testi...")
    edge_speak("Hazırım. Hey Groq diyerek başlayabilirsiniz.", out_id)
    print("  🎚️  Mikrofon kalibrasyonu (3 saniye sessiz olun)...")
    cal_audio = record_vad(in_id, max_secs=3, debug=True)
    print("  ✅ Kalibrasyon tamamlandı.\n")

    messages = [
        {"role": "system",
         "content": "Sen zeki, samimi ve bilgili bir sesli asistansın. "
                    "Türkçe konuş. Sorulara eksiksiz, doğal ve akıcı cevaplar ver. "
                    "Çok uzun paragraflar yerine akıcı kısa paragraflar kullan."}
    ]

    print("─" * 60)
    print("  👂 Wake-word bekleniyor... ('Hey Groq' deyin)")
    print("─" * 60)

    while True:
        try:
            # 1. Wake-word: VAD ile dinamik kayıt (eşiği düşür ki duyulsun)
            audio = record_vad(in_id, max_secs=4, debug=True)
            if audio.shape[0] < SAMPLE_RATE * 0.3:
                continue

            wav_file = audio_to_wav_file(audio)
            with open(wav_file, "rb") as f:
                result = groq_client.audio.transcriptions.create(
                    file=(wav_file, f.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )
            os.unlink(wav_file)
            heard = str(result).strip().lower()

            print(f"  🔍 Duyulan: '{heard}'")

            if not any(w in heard for w in WAKE_WORDS):
                print("  ↩️  Wake-word yok, tekrar dinleniyor...\n")
                continue

            # 2. Wake-word algılandı!
            print(f"\n\n  ✅ Wake-word → '{heard}'")
            print("─" * 60)

            # "Dinliyorum" sesini çalarken VAD ile kullanıcıyı dinlemeye başla
            t_speak = threading.Thread(
                target=edge_speak, args=("Dinliyorum.", out_id), daemon=True
            )
            t_speak.start()

            print("  🔴 Sorunuzu söyleyin...")
            cmd_audio = record_vad(in_id, max_secs=MAX_RECORD_SECS)
            t_speak.join()

            if cmd_audio.shape[0] < SAMPLE_RATE * 0.3:
                edge_speak("Duymadım, tekrar söyler misiniz?", out_id)
                continue

            # 3. Komutu transkribe et
            cmd_file = audio_to_wav_file(cmd_audio)
            with open(cmd_file, "rb") as f:
                cmd_result = groq_client.audio.transcriptions.create(
                    file=(cmd_file, f.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )
            os.unlink(cmd_file)
            user_text = str(cmd_result).strip()

            if not user_text:
                edge_speak("Anlayamadım, tekrar dener misiniz?", out_id)
                continue

            print(f"  🗣️  Siz  : \"{user_text}\"")
            messages.append({"role": "user", "content": user_text})

            # 4. LLM Streaming + Paralel TTS
            print("  🤖 Asistan:")
            full = stream_and_speak(groq_client, messages, out_id)
            messages.append({"role": "assistant", "content": full})

            print("\n" + "─" * 60)
            print("  👂 Dinleniyor... ('Hey Groq' deyin)")
            print("─" * 60)

        except KeyboardInterrupt:
            print("\n\n  👋 Görüşmek üzere!\n")
            break
        except Exception as e:
            print(f"\n  ⚠️ Hata: {e}")


if __name__ == "__main__":
    main()
