import os
import sys
import asyncio
import tempfile
import threading
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import resample
from groq import Groq
import edge_tts

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["COQUI_TOS_AGREED"] = "1"

# Türkçe Edge-TTS Sesi (AhmetNeural erkek, EmelNeural kadın)
EDGE_TTS_VOICE = "tr-TR-AhmetNeural"

TARGET_SAMPLE_RATE = 16000

def find_respeaker_devices():
    devices = sd.query_devices()
    input_id, output_id = None, None
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if any(k in name for k in ["respeaker", "uac1", "seeed"]):
            if dev['max_input_channels'] > 0 and input_id is None:
                input_id = i
            if dev['max_output_channels'] > 0 and output_id is None:
                output_id = i
    if input_id is None:
        input_id = sd.default.device[0]
    if output_id is None:
        output_id = sd.default.device[1]
    return input_id, output_id

def record_audio(duration, samplerate=16000, device_id=0):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, samplerate, audio)
    return tmp.name

async def edge_tts_synthesize(text, voice=EDGE_TTS_VOICE):
    """Edge-TTS ile metni sese çevirir ve WAV dosyası döner (< 300ms)"""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(tmp.name)
    return tmp.name

def play_audio_file(filepath, output_id):
    """MP3 veya WAV dosyasını ReSpeaker'a oynatır"""
    import subprocess
    # aplay yerine ffplay ile mp3 oynatma (daha evrensel)
    subprocess.run(
        ["ffplay", "-nodisp", "-autoexit", "-af", f"aresample={TARGET_SAMPLE_RATE}", filepath],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def synthesize_and_play(text, output_id):
    """Edge TTS ile sentezle ve anında çal"""
    if not text.strip():
        return
    mp3_file = asyncio.run(edge_tts_synthesize(text))
    play_audio_file(mp3_file, output_id)
    os.unlink(mp3_file)

def main():
    print("==================================================")
    print("⚡ Hey Groq! Ultra-Hızlı Asistan (Edge-TTS + Groq Streaming)")
    print("==================================================")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("🔑 Groq API Key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Mikrofon ID: [{input_id}] | 🎧 Kulaklık ID: [{output_id}]")
    print(f"🗣️ TTS Sesi: {EDGE_TTS_VOICE} (Microsoft Edge - Bulut Tabanlı Anlık Sentez)")
    print("\n✅ Hazır! 'Hey Groq' diyerek başlayın...\n")

    wake_words = ["hey groq", "hey grup", "hey krog", "hi groq", "groq", "grok", "hey grok"]

    messages_history = [
        {
            "role": "system",
            "content": "Sen zeki, samimi ve bilgili bir sesli asistansın. Türkçe konuş. Sorulara eksiksiz, doğal ve akıcı cevaplar ver."
        }
    ]

    while True:
        try:
            # 1. Sürekli Wake-Word Dinleme
            sys.stdout.write("\r👂 Dinleniyor... ('Hey Groq' deyin)        ")
            sys.stdout.flush()

            chunk_file = record_audio(duration=2.5, device_id=input_id)
            with open(chunk_file, "rb") as f:
                transcription = groq_client.audio.transcriptions.create(
                    file=(chunk_file, f.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )
            os.unlink(chunk_file)
            heard = str(transcription).strip().lower()

            if not heard:
                continue

            # 2. Wake-Word Algılama
            if not any(w in heard for w in wake_words):
                sys.stdout.write(f"\r🔍 '{heard}' - Wake-word yok, devam...     ")
                sys.stdout.flush()
                continue

            print(f"\n✨ Wake-word algılandı! Dinliyorum...")

            # 3. "Dinliyorum" Geri Bildirimi (Eşzamanlı)
            synthesize_and_play("Dinliyorum.", output_id)

            # 4. Kullanıcı Komudu Kaydet
            print("🔴 Sorunuzu söyleyin (5 saniye)...")
            cmd_file = record_audio(duration=5, device_id=input_id)
            with open(cmd_file, "rb") as f:
                cmd_trans = groq_client.audio.transcriptions.create(
                    file=(cmd_file, f.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )
            os.unlink(cmd_file)
            user_command = str(cmd_trans).strip()

            if not user_command:
                synthesize_and_play("Anlamadım, lütfen tekrar deneyin.", output_id)
                continue

            print(f"🗣️ Siz: \"{user_command}\"")
            messages_history.append({"role": "user", "content": user_command})

            # 5. Groq LLM Streaming + Eşzamanlı Edge-TTS
            stream = groq_client.chat.completions.create(
                messages=messages_history,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                stream=True
            )

            full_response = ""
            sentence_buffer = ""
            play_thread = None

            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                sentence_buffer += content

                # Cümle bitti mi? Varsa hemen arka planda sentezle & çal
                if any(p in sentence_buffer for p in ['.', '?', '!', '\n']):
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""

                    if sentence:
                        print(f"🤖 {sentence}")
                        # Önceki cümle hala çalıyorsa bitir
                        if play_thread and play_thread.is_alive():
                            play_thread.join()
                        # Yeni cümleyi arka planda çal (bir sonraki LLM parçası üretilirken)
                        play_thread = threading.Thread(
                            target=synthesize_and_play,
                            args=(sentence, output_id)
                        )
                        play_thread.start()

            # Kalan buffer
            if sentence_buffer.strip():
                if play_thread and play_thread.is_alive():
                    play_thread.join()
                synthesize_and_play(sentence_buffer.strip(), output_id)

            if play_thread and play_thread.is_alive():
                play_thread.join()

            messages_history.append({"role": "assistant", "content": full_response})
            print("\n--------------------------------------------------")

        except KeyboardInterrupt:
            print("\n👋 Görüşmek üzere!")
            break
        except Exception as e:
            print(f"\n⚠️ Hata: {e}")

if __name__ == "__main__":
    main()
