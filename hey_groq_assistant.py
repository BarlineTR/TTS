import os
import sys
import time
import torch
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import resample
from groq import Groq
from TTS.api import TTS

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

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
                
    if input_id is None:
        input_id = sd.default.device[0]
    if output_id is None:
        output_id = sd.default.device[1]
        
    return input_id, output_id

def record_stream_audio(duration=2.5, samplerate=16000, device_id=0):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
    sd.wait()
    wav.write("wakeword_input.wav", samplerate, audio)
    return "wakeword_input.wav"

def record_user_speech(duration=5, samplerate=16000, device_id=0):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device_id)
    sd.wait()
    wav.write("user_command.wav", samplerate, audio)
    return "user_command.wav"

def clean_text_for_tts(text):
    replacements = {
        'Ç': 'c', 'ç': 'c', 'Ğ': 'g', 'ğ': 'g', 'I': 'i', 'ı': 'i', 'İ': 'i',
        'Ö': 'o', 'ö': 'o', 'Ş': 's', 'ş': 's', 'Ü': 'u', 'ü': 'u',
        'Q': 'k', 'q': 'k', 'W': 'v', 'w': 'v', 'X': 'ks', 'x': 'ks'
    }
    text = text.lower()
    for old, new in replacements.items():
        text = text.replace(old.lower(), new)
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?"
    return "".join([c if c in allowed else ' ' for c in text])

def main():
    print("==================================================")
    print("🎙️ Wake-Word 'Hey Groq' Canlı Sesli Asistan (Görsel Loglu)")
    print("==================================================")
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key or groq_api_key == "gsk_SİZİN_GROQ_API_KEYİNİZ":
        groq_api_key = input("🔑 Groq API Anahtarınızı girin: ").strip()
        os.environ["GROQ_API_KEY"] = groq_api_key

    groq_client = Groq(api_key=groq_api_key)
    input_id, output_id = find_respeaker_devices()
    print(f"🎤 Mikrofon ID: [{input_id}] | 🎧 Kulaklık ID: [{output_id}]")
    
    print("\n⚡ TTS Modeli Yükleniyor...")
    tts = TTS(model_name="tts_models/tr/common-voice/glow-tts", progress_bar=False, gpu=False)
    print("✅ Asistan Hazır! ReSpeaker Mikrofonuna 'Hey Groq' Demenizi Bekliyor...\n")

    target_sample_rate = 16000
    messages_history = [
        {
            "role": "system",
            "content": "Sen zeki, samimi ve bilgili bir sesli asistansın. Türkçe konuş. Sorulara eksiksiz, doğal ve akıcı cevaplar ver."
        }
    ]

    wake_words = ["hey groq", "hey grup", "hey krog", "hi groq", "groq", "grup", "grok", "hey grok"]

    while True:
        try:
            # Arka plan dinlemesi görselleştirme
            sys.stdout.write("👂 [Ortam Dinleniyor... Sayın: 'Hey Groq']\r")
            sys.stdout.flush()

            chunk_file = record_stream_audio(duration=2.5, samplerate=16000, device_id=input_id)
            
            with open(chunk_file, "rb") as file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(chunk_file, file.read()),
                    model="whisper-large-v3",
                    language="tr",
                    response_format="text"
                )

            heard_text = str(transcription).strip().lower()
            if heard_text:
                print(f"\n🔍 Algılanan Ortam Sesi: \"{heard_text}\"")

            # Wake-word kontrolü
            if any(word in heard_text for word in wake_words):
                print(f"\n✨ WAKE-WORD ALGILANDI! (Duyulan: '{heard_text}')")
                print("🔊 'Dinliyorum seni'...")
                
                ack_wav = "ack.wav"
                tts.tts_to_file(text="dinliyorum seni", file_path=ack_wav)
                rate, data = wav.read(ack_wav)
                if rate != target_sample_rate:
                    num_samples = int(len(data) * target_sample_rate / rate)
                    data = resample(data, num_samples).astype(data.dtype)
                    rate = target_sample_rate
                sd.play(data, samplerate=rate, device=output_id)
                sd.wait()

                print("🔴 ŞİMDİ SORUNUZU SÖYLEYİN (5 saniye dinleniyor)...")
                cmd_file = record_user_speech(duration=5, samplerate=16000, device_id=input_id)
                with open(cmd_file, "rb") as file:
                    cmd_transcription = groq_client.audio.transcriptions.create(
                        file=(cmd_file, file.read()),
                        model="whisper-large-v3",
                        language="tr",
                        response_format="text"
                    )

                user_command = str(cmd_transcription).strip()
                print(f"🗣️ Komutunuz: \"{user_command}\"")

                if not user_command:
                    print("⚠️ Komut algılanamadı, dinlemeye dönülüyor.")
                    continue

                messages_history.append({"role": "user", "content": user_command})

                print("⚡ Groq Yanıt Üretiyor & Akış Yapılıyor...")
                stream = groq_client.chat.completions.create(
                    messages=messages_history,
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    stream=True
                )

                full_response = ""
                sentence_buffer = ""

                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    sentence_buffer += content

                    if any(p in sentence_buffer for p in ['.', '?', '!', '\n']):
                        clean_sentence = clean_text_for_tts(sentence_buffer.strip())
                        if clean_sentence.strip():
                            print(f"🤖 Asistan: {sentence_buffer.strip()}")
                            chunk_wav = "stream_chunk.wav"
                            tts.tts_to_file(text=clean_sentence, file_path=chunk_wav)
                            
                            rate, data = wav.read(chunk_wav)
                            if rate != target_sample_rate:
                                num_samples = int(len(data) * target_sample_rate / rate)
                                data = resample(data, num_samples).astype(data.dtype)
                                rate = target_sample_rate

                            sd.play(data, samplerate=rate, device=output_id)
                            sd.wait()

                        sentence_buffer = ""

                if sentence_buffer.strip():
                    clean_sentence = clean_text_for_tts(sentence_buffer.strip())
                    if clean_sentence.strip():
                        chunk_wav = "stream_chunk.wav"
                        tts.tts_to_file(text=clean_sentence, file_path=chunk_wav)
                        rate, data = wav.read(chunk_wav)
                        if rate != target_sample_rate:
                            num_samples = int(len(data) * target_sample_rate / rate)
                            data = resample(data, num_samples).astype(data.dtype)
                            rate = target_sample_rate
                        sd.play(data, samplerate=rate, device=output_id)
                        sd.wait()

                messages_history.append({"role": "assistant", "content": full_response})
                print("\n--------------------------------------------------")

        except Exception as e:
            print(f"⚠️ Hata: {e}")

if __name__ == "__main__":
    main()
