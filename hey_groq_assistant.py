import os
import sys
import asyncio
import tempfile
import threading
import subprocess
import queue
import time
import re

import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from groq import Groq
import edge_tts

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["COQUI_TOS_AGREED"] = "1"

# ─── Ayarlar ────────────────────────────────────────────────────────────────
EDGE_TTS_VOICE = "tr-TR-AhmetNeural"
SAMPLE_RATE = 16000
WAKE_WORDS = [
    "hey groq", "hey grok", "hey grup", "hey krog", "hey crock",
    "a groq", "a grok", "groq", "grok", "hi groq"
]
VAD_THRESHOLD = 500
VAD_SILENCE_SECS = 0.75
CHUNK_SECS = 0.06
MAX_RECORD_SECS = 10
CONVO_TIMEOUT_SECS = 15
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.55
LLM_MAX_TOKENS = 300
TTS_MIN_CHARS = 35
TTS_MAX_CHARS = 180

# ─── Audio ──────────────────────────────────────────────────────────────────
audio_queue = queue.Queue(maxsize=32)
audio_stop_event = threading.Event()
audio_generation_lock = threading.Lock()
current_audio_generation = 0


def next_audio_generation():
    global current_audio_generation
    with audio_generation_lock:
        current_audio_generation += 1
        generation = current_audio_generation
    try:
        while True:
            audio_queue.get_nowait()
    except queue.Empty:
        pass
    return generation


def stop_audio():
    sd.stop()
    next_audio_generation()


def audio_player_worker(out_id):
    while not audio_stop_event.is_set():
        try:
            item = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            continue
        generation, sample_rate, pcm = item
        with audio_generation_lock:
            if generation != current_audio_generation:
                continue
        try:
            sd.play(pcm, samplerate=sample_rate, device=out_id, blocking=True)
        except Exception as exc:
            print(f"\n  ⚠️ Audio playback: {exc}")


# ─── Cihaz ──────────────────────────────────────────────────────────────────
def find_respeaker():
    devices = sd.query_devices()
    in_id, out_id = None, None
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if any(k in name for k in ["respeaker", "uac1", "seeed"]):
            if dev["max_input_channels"] > 0 and in_id is None:
                in_id = i
            if dev["max_output_channels"] > 0 and out_id is None:
                out_id = i
    in_id = in_id if in_id is not None else sd.default.device[0]
    out_id = out_id if out_id is not None else sd.default.device[1]
    return in_id, out_id


# ─── VAD ────────────────────────────────────────────────────────────────────
def record_vad(device_id, max_secs=MAX_RECORD_SECS, debug=False,
               wait_for_speech=True) -> np.ndarray:
    chunk = int(SAMPLE_RATE * CHUNK_SECS)
    frames, silent_chunks, speaking = [], 0, False
    max_chunks = int(max_secs / CHUNK_SECS)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        device=device_id, blocksize=chunk) as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk)
            rms = int(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            if debug:
                bar = "█" * min(int(rms / 80), 25)
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
        return np.zeros((0, 1), dtype="int16")
    return np.concatenate(frames, axis=0)


def audio_to_wav_file(audio_np: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav.write(tmp.name, SAMPLE_RATE, audio_np)
    return tmp.name


# ─── STT ────────────────────────────────────────────────────────────────────
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
        try:
            os.unlink(wf)
        except OSError:
            pass


# ─── Streaming TTS ──────────────────────────────────────────────────────────
async def _edge_stream_to_queue(text, generation):
    """Edge-TTS audio stream -> FFmpeg stdin -> PCM stdout -> RAM queue."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-loglevel", "error",
        "-f", "mp3", "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)

    async def write_audio():
        try:
            async for message in communicate.stream():
                if message["type"] == "audio":
                    proc.stdin.write(message["data"])
                    await proc.stdin.drain()
            proc.stdin.close()
            await proc.stdin.wait_closed()
        except Exception:
            try:
                proc.stdin.close()
            except Exception:
                pass
            raise

    writer_task = asyncio.create_task(write_audio())
    bytes_per_chunk = int(SAMPLE_RATE * 0.06 * 2)
    try:
        while True:
            data = await proc.stdout.read(bytes_per_chunk)
            if not data:
                break
            pcm = np.frombuffer(data, dtype=np.int16).copy()
            with audio_generation_lock:
                if generation != current_audio_generation:
                    break
            await asyncio.to_thread(audio_queue.put, (generation, SAMPLE_RATE, pcm))
        await writer_task
        await proc.wait()
    finally:
        if not writer_task.done():
            writer_task.cancel()
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


def speak_streaming(text: str, out_id: int, generation=None):
    if not text.strip():
        return
    if generation is None:
        generation = next_audio_generation()
    try:
        asyncio.run(_edge_stream_to_queue(text.strip(), generation))
    except Exception as exc:
        print(f"\n  ⚠️ TTS streaming hatası: {exc}")


def speak_once(text: str, out_id: int):
    generation = next_audio_generation()
    speak_streaming(text, out_id, generation)


# ─── Cümle buffer ───────────────────────────────────────────────────────────
def extract_tts_sentences(buffer: str, final=False):
    ready = []
    buffer = buffer.replace("\n\n", " ")
    buffer = re.sub(r"\s+", " ", buffer)
    while True:
        match = re.search(r"^(.{%d,}?[.!?…]+)(?:\s+|$)" % TTS_MIN_CHARS, buffer)
        if not match:
            break
        sentence = match.group(1).strip()
        if sentence:
            ready.append(sentence)
        buffer = buffer[match.end():].lstrip()
    if len(buffer) >= TTS_MAX_CHARS:
        cut = max(buffer.rfind(",", 0, TTS_MAX_CHARS),
                   buffer.rfind(" ", 0, TTS_MAX_CHARS))
        if cut >= TTS_MIN_CHARS:
            ready.append(buffer[:cut].strip())
            buffer = buffer[cut:].lstrip()
    if final and buffer.strip():
        ready.append(buffer.strip())
        buffer = ""
    return ready, buffer


# ─── LLM + TTS Pipeline ─────────────────────────────────────────────────────
def stream_and_speak_pipeline(groq_client, messages, out_id):
    """LLM token stream -> sentence queue -> streaming TTS -> audio queue."""
    full_response = ""
    text_buffer = ""
    generation = next_audio_generation()
    tts_queue = queue.Queue(maxsize=4)

    def tts_worker():
        try:
            while True:
                sentence = tts_queue.get()
                if sentence is None:
                    break
                print(f"\n  🔊 TTS: {sentence}")
                speak_streaming(sentence, out_id, generation)
        except Exception as exc:
            print(f"\n  ⚠️ TTS worker: {exc}")

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    stream = groq_client.chat.completions.create(
        messages=messages,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        stream=True
    )

    sys.stdout.write("  🤖 ")
    sys.stdout.flush()
    try:
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            full_response += token
            text_buffer += token
            sys.stdout.write(token)
            sys.stdout.flush()
            sentences, text_buffer = extract_tts_sentences(text_buffer)
            for sentence in sentences:
                tts_queue.put(sentence)
        sentences, text_buffer = extract_tts_sentences(text_buffer, final=True)
        for sentence in sentences:
            tts_queue.put(sentence)
    finally:
        tts_queue.put(None)
        tts_thread.join()
    print()
    return full_response


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\033[94m" + r"""
  ██╗  ██╗███████╗██╗   ██╗     ██████╗ ██████╗  ██████╗  ██████╗
  ██║  ██║██╔════╝╚██╗ ██╔╝    ██╔════╝ ██╔══██╗██╔═══██╗██╔═══██╗
  ███████║█████╗   ╚████╔╝     ██║  ███╗██████╔╝██║   ██║██║   ██║
  ██╔══██║██╔══╝    ╚██╔╝      ██║   ██║██████╔╝██║   ██║██║▄▄ ██║
  ██║  ██║███████╗   ██║       ╚██████╔╝██╔══██╗╚██████╔╝╚██████╔╝
  ╚═╝  ╚═╝╚══════╝   ╚═╝        ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝
   Jetson Gerçek Zamanlı Sesli Asistan
   Groq Streaming + Sentence TTS + Audio Queue
    """ + "\033[0m")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("  🔑 Groq API Key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    in_id, out_id = find_respeaker()
    print(f"  🎤 Mikrofon  : [{in_id}]")
    print(f"  🎧 Kulaklık  : [{out_id}]")
    print(f"  🗣️  TTS       : {EDGE_TTS_VOICE} (Streaming)")
    print(f"  🧠 LLM       : {LLM_MODEL}")
    print(f"  ⏱️  Timeout   : {CONVO_TIMEOUT_SECS}s\n")

    player_thread = threading.Thread(target=audio_player_worker, args=(out_id,), daemon=True)
    player_thread.start()

    print("  🔊 Başlangıç sesi...")
    speak_once("Hazırım. Hey Groq diyerek başlayın.", out_id)

    messages = [{
        "role": "system",
        "content": (
            "Sen hızlı ve doğal konuşan bir Türkçe sesli asistansın. "
            "Cevaplarını konuşmaya uygun üret. Gereksiz uzun açıklamalar yapma. "
            "Önce doğrudan sonucu söyle, gerekirse kısa açıklama ekle. "
            "Markdown, tablo, kod bloğu ve emoji kullanma."
        )
    }]

    in_conversation = False
    last_interaction = 0

    while True:
        try:
            if not in_conversation:
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
                print("\n  ✅ Wake-word algılandı!")
                in_conversation = True
                last_interaction = time.time()
                t_ack = threading.Thread(target=speak_once, args=("Dinliyorum.", out_id), daemon=True)
                t_ack.start()
                print("  🔴 Sorunuzu söyleyin...")
                cmd_audio = record_vad(in_id, max_secs=MAX_RECORD_SECS)
            else:
                elapsed = time.time() - last_interaction
                remaining = CONVO_TIMEOUT_SECS - elapsed
                if remaining <= 0:
                    print("\n  🟡 Sohbet zaman aşımı → Wake-word moduna dönülüyor.")
                    in_conversation = False
                    continue
                print(f"\n  🟢 [Sohbet Modu] Konuşun... ({remaining:.0f}s kaldı)")
                cmd_audio = record_vad(in_id, max_secs=MAX_RECORD_SECS, debug=True)

            if cmd_audio.shape[0] < SAMPLE_RATE * 0.5:
                continue

            user_text = transcribe(groq_client, cmd_audio)
            if not user_text:
                continue

            print(f'\n  🗣️  Siz: "{user_text}"')
            messages.append({"role": "user", "content": user_text})
            last_interaction = time.time()
            print("  🤖 Asistan:")
            full = stream_and_speak_pipeline(groq_client, messages, out_id)
            if full.strip():
                messages.append({"role": "assistant", "content": full})
            last_interaction = time.time()
            in_conversation = True

        except KeyboardInterrupt:
            print("\n\n  👋 Görüşmek üzere!\n")
            break
        except Exception as exc:
            print(f"\n  ⚠️ Hata: {exc}")

    audio_stop_event.set()
    stop_audio()


if __name__ == "__main__":
    main()
