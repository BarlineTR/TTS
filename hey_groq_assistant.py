import os
import sys
import asyncio
import tempfile
import threading
import subprocess
import queue
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from groq import Groq
import edge_tts

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["COQUI_TOS_AGREED"] = "1"

# ─── Ayarlar ─────────────────────────────────────────────────────────────────
EDGE_TTS_VOICE       = "tr-TR-AhmetNeural"
SAMPLE_RATE          = 16000
WAKE_WORDS           = [
    "hey groq", "hey grok", "hey grup", "hey krog", "hey crock",
    "a groq", "a grok", "groq", "grok", "hi groq"
]
VAD_THRESHOLD        = 500    # Ses eşiği (kalibrasyondan ayarla)
VAD_SILENCE_SECS     = 1.2   # Sessizlik → konuşma bitti
CHUNK_SECS           = 0.08  # Streaming blok (80ms)
MAX_RECORD_SECS      = 10
CONVO_TIMEOUT_SECS   = 15    # Sohbet modunda bekleme süresi (sonra wake-word'e döner)
# ─────────────────────────────────────────────────────────────────────────────


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
    in_id  = in_id  if in_id  is not None else sd.default.device[0]
    out_id = out_id if out_id is not None else sd.default.device[1]
    return in_id, out_id


def record_vad(device_id, max_secs=MAX_RECORD_SECS,
               debug=False, wait_for_speech=True) -> np.ndarray:
    """
    VAD tabanlı dinamik kayıt.
    wait_for_speech=False → sadece var olan sesi yakala (timeout yoksa boş döner)
    """
    chunk = int(SAMPLE_RATE * CHUNK_SECS)
    frames, silent_chunks, speaking = [], 0, False
    max_chunks = int(max_secs / CHUNK_SECS)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                        device=device_id, blocksize=chunk) as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk)
            rms = int(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

            if debug:
                bar = '█' * min(int(rms / 80), 25)
                sys.stdout.write(f"  🎙️  [{bar:<25}] {rms:4d}\r")
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
            elif not wait_for_speech:
                break

    if not frames:
        return np.zeros((0, 1), dtype='int16')
    return np.concatenate(frames, axis=0)


def audio_to_wav_file(audio_np: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, audio_np)
    return tmp.name


# ─── Tek Çağrı TTS (Maksimum Akıcılık) ──────────────────────────────────────
def _edge_synthesize_full(text: str, out_id: int):
    """
    Tüm yanıtı tek bir Edge-TTS çağrısıyla sentezler.
    Tek ağ isteği → doğal prozodi → sıfır boşluk.
    """
    mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    wf  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    mp3.close(); wf.close()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def _s():
        await edge_tts.Communicate(text, EDGE_TTS_VOICE).save(mp3.name)
    loop.run_until_complete(_s()); loop.close()
    subprocess.run(["ffmpeg", "-y", "-i", mp3.name,
                    "-ar", str(SAMPLE_RATE), "-ac", "1", wf.name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rate, data = wav.read(wf.name)
    os.unlink(mp3.name); os.unlink(wf.name)
    sd.play(data, samplerate=rate, device=out_id)
    sd.wait()


def stream_and_speak_pipeline(groq_client, messages, out_id):
    """
    1. LLM akışıyla tüm yanıtı topla (metin terminale canlı yazılır)
    2. Tamamlanan yanıtı tek Edge-TTS çağrısıyla seslendir → boşluksuz, akıcı
    """
    full_response = ""

    stream = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        stream=True
    )

    sys.stdout.write("  🤖 ")
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_response += token
        sys.stdout.write(token)
        sys.stdout.flush()
    print()  # Yeni satır

    # Tüm yanıtı tek seferde seslendir (maksimum akıcılık)
    if full_response.strip():
        _edge_synthesize_full(full_response.strip(), out_id)

    return full_response
# ─────────────────────────────────────────────────────────────────────────────


def transcribe(groq_client, audio_np: np.ndarray) -> str:
    if audio_np.shape[0] < SAMPLE_RATE * 0.3:
        return ""
    wf = audio_to_wav_file(audio_np)
    try:
        with open(wf, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                file=(wf, f.read()),
                model="whisper-large-v3",
                language="tr",
                response_format="text"
            )
        return str(result).strip()
    finally:
        os.unlink(wf)


def edge_speak_once(text: str, out_id: int):
    """Tek cümle için hızlı TTS (onay sesleri için)"""
    mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    wf  = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    mp3.close(); wf.close()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def _s():
        await edge_tts.Communicate(text, EDGE_TTS_VOICE).save(mp3.name)
    loop.run_until_complete(_s()); loop.close()
    subprocess.run(["ffmpeg", "-y", "-i", mp3.name, "-ar", str(SAMPLE_RATE),
                    "-ac", "1", wf.name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rate, data = wav.read(wf.name)
    os.unlink(mp3.name); os.unlink(wf.name)
    sd.play(data, samplerate=rate, device=out_id); sd.wait()


def main():
    print("\033[94m" + r"""
  ██╗  ██╗███████╗██╗   ██╗     ██████╗ ██████╗  ██████╗  ██████╗
  ██║  ██║██╔════╝╚██╗ ██╔╝    ██╔════╝ ██╔══██╗██╔═══██╗██╔═══██╗
  ███████║█████╗   ╚████╔╝     ██║  ███╗██████╔╝██║   ██║██║   ██║
  ██╔══██║██╔══╝    ╚██╔╝      ██║   ██║██╔══██╗██║   ██║██║▄▄ ██║
  ██║  ██║███████╗   ██║       ╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝
  ╚═╝  ╚═╝╚══════╝   ╚═╝        ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝
  Jetson Sesli Asistan │ Pipeline TTS + Sohbet Modu
    """ + "\033[0m")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("  🔑 Groq API Key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    in_id, out_id = find_respeaker()
    print(f"  🎤 Mikrofon  : [{in_id}]  |  🎧 Kulaklık : [{out_id}]")
    print(f"  🗣️  TTS       : {EDGE_TTS_VOICE} (Pipeline Mode)")
    print(f"  ⏱️  Sohbet     : {CONVO_TIMEOUT_SECS}s sessizlik → wake-word moduna dön\n")

    print("  🔊 Başlangıç sesi...")
    edge_speak_once("Hazırım. Hey Groq diyerek başlayın.", out_id)

    messages = [
        {"role": "system",
         "content": "Sen zeki, samimi ve bilgili bir sesli asistansın. "
                    "Türkçe konuş. Sorulara eksiksiz, doğal ve akıcı cevaplar ver."}
    ]

    # ── Ana Döngü ──────────────────────────────────────────────────────────
    in_conversation = False   # Sohbet modunda mıyız?
    last_interaction = 0      # Son etkileşim zamanı

    while True:
        try:
            if not in_conversation:
                # ── WAKE-WORD MODU ──────────────────────────────────────
                print("\n" + "─" * 60)
                print("  🔵 [Wake-Word Modu] 'Hey Groq' deyin...")
                print("─" * 60)

                audio = record_vad(in_id, max_secs=5, debug=True)
                heard = transcribe(groq_client, audio)
                if not heard:
                    continue
                print(f"  🔍 Duyulan: '{heard}'")
                if not any(w in heard.lower() for w in WAKE_WORDS):
                    print("  ↩️  Wake-word yok.\n")
                    continue

                print(f"\n  ✅ Wake-word algılandı!")
                in_conversation = True
                last_interaction = time.time()

                # "Dinliyorum" arka planda çalsın, komut dinlensin
                t_ack = threading.Thread(
                    target=edge_speak_once, args=("Dinliyorum.", out_id), daemon=True
                )
                t_ack.start()
                print("  🔴 Sorunuzu söyleyin...")
                cmd_audio = record_vad(in_id, max_secs=MAX_RECORD_SECS)
                t_ack.join()

            else:
                # ── SOHBET MODU ─────────────────────────────────────────
                elapsed = time.time() - last_interaction
                remaining = CONVO_TIMEOUT_SECS - elapsed

                if remaining <= 0:
                    print("\n  🟡 Sohbet zaman aşımı → Wake-word moduna dönülüyor.")
                    in_conversation = False
                    continue

                print(f"\n  🟢 [Sohbet Modu] Konuşun... ({remaining:.0f}s kaldı)")
                cmd_audio = record_vad(in_id, max_secs=MAX_RECORD_SECS, debug=True)

            # ── Komutu Transkribe Et ──────────────────────────────────────
            if cmd_audio.shape[0] < SAMPLE_RATE * 0.5:
                if in_conversation:
                    continue  # Boş → tekrar dinle (timeout düşer)
                continue

            user_text = transcribe(groq_client, cmd_audio)
            if not user_text:
                continue

            print(f"\n  🗣️  Siz    : \"{user_text}\"")
            messages.append({"role": "user", "content": user_text})
            last_interaction = time.time()

            # ── LLM + Pipeline TTS ───────────────────────────────────────
            print("  🤖 Asistan:")
            full = stream_and_speak_pipeline(groq_client, messages, out_id)
            messages.append({"role": "assistant", "content": full})
            last_interaction = time.time()  # Cevap sonrası da saat sıfırla
            in_conversation = True          # Cevaptan sonra sohbet modu devam

        except KeyboardInterrupt:
            print("\n\n  👋 Görüşmek üzere!\n")
            break
        except Exception as e:
            print(f"\n  ⚠️ Hata: {e}")


if __name__ == "__main__":
    main()
