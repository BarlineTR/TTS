import os
import sys
import asyncio
import tempfile
import threading
import subprocess
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import resample
from groq import Groq
import edge_tts

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["COQUI_TOS_AGREED"] = "1"

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
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=1, dtype='int16', device=device_id)
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, samplerate, audio)
    return tmp.name


def edge_synthesize(text, voice=EDGE_TTS_VOICE):
    """Edge-TTS ile sentezle, MP3'ü WAV'a çevir, numpy array döndür"""
    mp3_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    mp3_tmp.close()
    wav_tmp.close()

    # Yeni event loop oluştur (threading içinde asyncio güvenli kullanım)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _synth():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(mp3_tmp.name)

    loop.run_until_complete(_synth())
    loop.close()

    # MP3 → WAV (ffmpeg ile)
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_tmp.name,
         "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1", wav_tmp.name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    rate, data = wav.read(wav_tmp.name)
    os.unlink(mp3_tmp.name)
    os.unlink(wav_tmp.name)
    return rate, data


def speak(text, output_id):
    """Metni sentezle ve ReSpeaker kulaklığından çal"""
    if not text.strip():
        return
    try:
        rate, data = edge_synthesize(text)
        sd.play(data, samplerate=rate, device=output_id)
        sd.wait()
    except Exception as e:
        print(f"⚠️ TTS Hatası: {e}")


def main():
    print("==================================================")
    print("⚡ Hey Groq! Ultra-Hızlı Asistan v3 (Edge-TTS + Groq Streaming)")
    print("==================================================")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("🔑 Groq API Key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Mikrofon ID: [{input_id}] | 🎧 Kulaklık ID: [{output_id}]")
    print(f"🗣️ TTS: {EDGE_TTS_VOICE} (Microsoft Edge)")

    # TTS testi - ses çıkışını doğrula
    print("\n🔊 Ses testi yapılıyor...")
    speak("Sistem hazır. Hey Groq diyerek başlayabilirsiniz.", output_id)
    print("✅ Ses testi tamamlandı! 'Hey Groq' diyerek başlayın.\n")

    wake_words = [
        "hey groq", "hey grup", "hey krog", "hi groq",
        "groq", "grok", "hey grok", "heygroq", "a groq", "a grok"
    ]

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

            if heard:
                sys.stdout.write(f"\r🔍 Duyulan: '{heard}'                              ")
                sys.stdout.flush()

            # 2. Wake-Word Kontrolü
            if not any(w in heard for w in wake_words):
                continue

            print(f"\n✨ Wake-word algılandı! ('{heard}')")

            # 3. Onay sesi (arka planda sentezlerken kayıt başlasın)
            print("🔴 Sorunuzu söyleyin (5 saniye)...")
            ack_thread = threading.Thread(target=speak, args=("Dinliyorum.", output_id))
            ack_thread.start()

            # "Dinliyorum" çalarken kullanıcı komutunu da kaydet
            cmd_file = record_audio(duration=5, device_id=input_id)
            ack_thread.join()  # Onay sesi bitmeden çakışma olmasın

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
                speak("Anlamadım, tekrar dener misiniz?", output_id)
                continue

            print(f"🗣️ Siz: \"{user_command}\"")
            messages_history.append({"role": "user", "content": user_command})

            # 4. Groq LLM Streaming + Paralel Edge-TTS
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

                if any(p in sentence_buffer for p in ['.', '?', '!', '\n']):
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""
                    if sentence:
                        print(f"🤖 {sentence}")
                        # Önceki cümle bitsin
                        if play_thread and play_thread.is_alive():
                            play_thread.join()
                        # Yeni cümleyi arka planda çal
                        play_thread = threading.Thread(
                            target=speak, args=(sentence, output_id)
                        )
                        play_thread.start()

            if sentence_buffer.strip():
                if play_thread and play_thread.is_alive():
                    play_thread.join()
                speak(sentence_buffer.strip(), output_id)

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
