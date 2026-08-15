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
EDGE_TTS_RATE = "+20%"
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
TTS_MIN_CHARS = 70
TTS_MAX_CHARS = 260
TTS_PREFETCH = 3

# ─── Audio ──────────────────────────────────────────────────────────────────
# Ses çıkışını güvenilir tutuyoruz. Önceki sürümde MP3 pipe -> FFmpeg ->
# PCM queue zinciri Jetson'daki bazı FFmpeg/PortAudio kombinasyonlarında
# ses üretimini susturabiliyordu.
audio_stop_event = threading.Event()
audio_generation_lock = threading.Lock()
current_audio_generation = 0


def next_audio_generation():
    global current_audio_generation

    with audio_generation_lock:
        current_audio_generation += 1
        return current_audio_generation


def stop_audio():
    sd.stop()
    next_audio_generation()


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
def _edge_synthesize_sentence(text: str, out_id: int, generation: int):
    """
    Tüm cevabı değil, yalnızca hazır cümleyi TTS'ye gönderir.

    Böylece LLM:
        cümle 1 -> TTS -> ses
        cümle 2 -> TTS -> ses
        ...

    şeklinde çalışır.

    Burada Edge-TTS -> MP3 -> FFmpeg -> WAV -> sounddevice zincirini
    koruyoruz. Bu, önceki pipe/PCM streaming sürümüne göre Jetson üzerinde
    çok daha güvenilir ses çıkışı sağlar.
    """
    if not text.strip():
        return

    mp3_path = None
    wav_path = None

    try:
        mp3_file = tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False
        )
        wav_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        mp3_path = mp3_file.name
        wav_path = wav_file.name

        mp3_file.close()
        wav_file.close()

        loop = asyncio.new_event_loop()

        try:
            asyncio.set_event_loop(loop)

            async def synthesize():
                communicate = edge_tts.Communicate(
                    text,
                    EDGE_TTS_VOICE,
                    rate=EDGE_TTS_RATE
                )
                await communicate.save(mp3_path)

            loop.run_until_complete(synthesize())

        finally:
            loop.close()

        with audio_generation_lock:
            if generation != current_audio_generation:
                return

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-i", mp3_path,
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-f", "wav",
                wav_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            error_text = result.stderr.decode(
                "utf-8",
                errors="replace"
            ).strip()
            raise RuntimeError(
                f"FFmpeg TTS decode başarısız: {error_text}"
            )

        rate, data = wav.read(wav_path)

        if data is None or len(data) == 0:
            raise RuntimeError("TTS WAV verisi boş geldi.")

        with audio_generation_lock:
            if generation != current_audio_generation:
                return

        if data.dtype != np.int16:
            data = data.astype(np.int16)

        sd.play(
            data,
            samplerate=rate,
            device=out_id,
            blocking=True
        )

    except Exception as exc:
        print(f"\n  ⚠️ TTS/playback hatası: {exc}")

    finally:
        for path in (mp3_path, wav_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def speak_streaming(text: str, out_id: int, generation=None):
    if not text.strip():
        return

    if generation is None:
        generation = next_audio_generation()

    _edge_synthesize_sentence(
        text.strip(),
        out_id,
        generation
    )


def speak_once(text: str, out_id: int):
    generation = next_audio_generation()
    speak_streaming(text, out_id, generation)


# ─── TTS metin temizleme ─────────────────────────────────────────────────────
# LLM'nin ürettiği emoji, markdown ve TTS'nin gereksiz yere seslendireceği
# biçimlendirme karakterlerini temizler.
EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)

def clean_tts_text(text: str) -> str:
    """TTS'ye gitmeden önce emoji/markdown ve gereksiz durakları temizle."""
    if not text:
        return ""

    # Emoji'leri kaldır.
    text = EMOJI_RE.sub("", text)

    # Markdown kalıntıları.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"[*_#~]+", " ", text)

    # Emoji temizliğinden sonra oluşan boşlukları toparla.
    text = re.sub(r"\s+", " ", text).strip()

    # TTS'nin gereksiz uzun duraklamasına neden olabilecek noktalama.
    # Noktayı tamamen silmiyoruz, çünkü cümle yapısını korumak gerekiyor.
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


# ─── Cümle buffer ───────────────────────────────────────────────────────────
def extract_tts_sentences(buffer: str, final=False):
    """
    TTS için küçük cümleler yerine daha büyük doğal konuşma parçaları üretir.

    Amaç:
      "İyiyim, teşekkür ederim. Sana nasıl yardımcı olabilirim?"
          -> tek TTS parçası

      Uzun cevaplarda ise yaklaşık 2-3 cümlelik parçalar oluşturmak.

    Böylece her nokta için yeni bir Edge-TTS isteği yapılmaz.
    """
    ready = []
    buffer = re.sub(r"\s+", " ", buffer).strip()

    # Güçlü bitişleri kontrol et. Ancak kısa parçayı hemen göndermiyoruz.
    while True:
        matches = list(re.finditer(r"[.!?]+(?:\s+|$)", buffer))
        if not matches:
            break

        # En az TTS_MIN_CHARS olacak şekilde en uygun cümle sonunu bul.
        chosen = None
        for m in matches:
            candidate = buffer[:m.end()].strip()
            if len(candidate) >= TTS_MIN_CHARS:
                chosen = m
                break

        if chosen is None:
            break

        candidate = clean_tts_text(buffer[:chosen.end()].strip())
        if candidate:
            ready.append(candidate)

        buffer = buffer[chosen.end():].lstrip()

        # İlk parça yeterince büyükse ve kalan metin de varsa,
        # her cümleyi ayrı ayrı göndermeyelim. Sonraki cümleleri
        # mümkün olduğunca aynı buffer'da tutuyoruz.
        if len(ready) >= 1 and len(buffer) < TTS_MAX_CHARS:
            break

    # Buffer fazla büyürse doğal bir sınırdan böl.
    if len(buffer) >= TTS_MAX_CHARS:
        cut_candidates = [
            buffer.rfind(". ", 0, TTS_MAX_CHARS),
            buffer.rfind("! ", 0, TTS_MAX_CHARS),
            buffer.rfind("? ", 0, TTS_MAX_CHARS),
            buffer.rfind(", ", 0, TTS_MAX_CHARS),
            buffer.rfind(" ", 0, TTS_MAX_CHARS),
        ]

        cut = max(cut_candidates)

        if cut >= TTS_MIN_CHARS:
            # Noktalama sınırında ise noktalamayı da dahil et.
            if buffer[cut] in ".!?":
                cut += 1

            sentence = clean_tts_text(buffer[:cut])
            if sentence:
                ready.append(sentence)

            buffer = buffer[cut:].lstrip()

    if final and buffer:
        sentence = clean_tts_text(buffer)
        if sentence:
            ready.append(sentence)
        buffer = ""

    return ready, buffer


# ─── LLM + TTS Pipeline ─────────────────────────────────────────────────────
def stream_and_speak_pipeline(groq_client, messages, out_id):
    """
    Gerçek zamanlı LLM + paralel TTS pipeline.

    Önceki sürüm:
        LLM -> TTS 1 -> PLAY 1 -> TTS 2 -> PLAY 2 -> ...

    Yeni sürüm:
        LLM -> TTS 1 ───────────────┐
             TTS 2 (prefetch) ─────┤
             TTS 3 (prefetch) ─────┤
                                    ↓
                              PLAY 1 -> PLAY 2 -> PLAY 3

    Böylece kullanıcı birinci parçayı dinlerken sonraki parçalar
    arka planda hazırlanır.
    """
    from concurrent.futures import ThreadPoolExecutor

    full_response = ""
    text_buffer = ""
    generation = next_audio_generation()

    # Aynı anda birkaç TTS isteğini hazırlıyoruz.
    executor = ThreadPoolExecutor(
        max_workers=TTS_PREFETCH,
        thread_name_prefix="tts-prefetch"
    )

    futures = []

    def synthesize_to_wav(sentence, index):
        """TTS sentezle, WAV dosyasını hazırla. Playback burada yapılmaz."""
        mp3_path = None
        wav_path = None

        try:
            mp3_file = tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False
            )
            wav_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            )

            mp3_path = mp3_file.name
            wav_path = wav_file.name

            mp3_file.close()
            wav_file.close()

            loop = asyncio.new_event_loop()

            try:
                asyncio.set_event_loop(loop)

                async def synthesize():
                    communicate = edge_tts.Communicate(
                        sentence,
                        EDGE_TTS_VOICE,
                        rate=EDGE_TTS_RATE
                    )
                    await communicate.save(mp3_path)

                loop.run_until_complete(synthesize())

            finally:
                loop.close()

            with audio_generation_lock:
                if generation != current_audio_generation:
                    return None

            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "error",
                    "-i", mp3_path,
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    "-f", "wav",
                    wav_path
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )

            if result.returncode != 0:
                error_text = result.stderr.decode(
                    "utf-8",
                    errors="replace"
                ).strip()
                raise RuntimeError(
                    f"FFmpeg TTS decode başarısız: {error_text}"
                )

            rate, data = wav.read(wav_path)

            if data is None or len(data) == 0:
                raise RuntimeError("TTS WAV verisi boş geldi.")

            if data.dtype != np.int16:
                data = data.astype(np.int16)

            return rate, data.copy()

        finally:
            for path in (mp3_path, wav_path):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def submit_sentence(sentence):
        sentence = clean_tts_text(sentence)

        if not sentence:
            return

        index = len(futures)
        print(f"\n  🔊 TTS hazırlanıyor #{index + 1}: {sentence}")

        future = executor.submit(
            synthesize_to_wav,
            sentence,
            index
        )

        futures.append((index, sentence, future))

    try:
        stream = groq_client.chat.completions.create(
            messages=messages,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True
        )

        sys.stdout.write("  🤖 ")
        sys.stdout.flush()

        for chunk in stream:
            token = chunk.choices[0].delta.content or ""

            if not token:
                continue

            full_response += token
            text_buffer += token

            # Terminal çıktısını bozmadan göster.
            sys.stdout.write(token)
            sys.stdout.flush()

            sentences, text_buffer = extract_tts_sentences(text_buffer)

            for sentence in sentences:
                submit_sentence(sentence)

        # Son parçayı al.
        sentences, text_buffer = extract_tts_sentences(
            text_buffer,
            final=True
        )

        for sentence in sentences:
            submit_sentence(sentence)

        print()

        # TTS'ler sırasını koruyarak oynatılır.
        for index, sentence, future in futures:
            try:
                with audio_generation_lock:
                    if generation != current_audio_generation:
                        break

                rate, data = future.result()

                with audio_generation_lock:
                    if generation != current_audio_generation:
                        break

                print(f"  🔊 Oynatılıyor #{index + 1}: {sentence}")

                sd.play(
                    data,
                    samplerate=rate,
                    device=out_id,
                    blocking=True
                )

            except Exception as exc:
                print(f"\n  ⚠️ TTS #{index + 1} hatası: {exc}")

    finally:
        executor.shutdown(wait=True, cancel_futures=True)

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

    print("  🔊 Başlangıç sesi...")
    speak_once("Hazırım. Hey Groq diyerek başlayın.", out_id)

    messages = [{
        "role": "system",
        "content": (
            "Sen hızlı ve doğal konuşan bir Türkçe sesli asistansın. "
            "Cevaplarını konuşmaya uygun üret. Gereksiz uzun açıklamalar yapma. "
            "Önce doğrudan sonucu söyle, gerekirse kısa açıklama ekle. "
            "Markdown, tablo, kod bloğu ve emoji kullanma. "
                "Emoji, sembol veya yüz ifadesi üretme; konuşma için yalnızca "
                "normal Türkçe metin kullan."
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
                print("  🔴 Buyurun efendim...")
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