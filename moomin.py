import sounddevice as sd
import numpy as np
import whisper
import pyttsx3
import requests
import tempfile
import os
import scipy.io.wavfile

# === CONFIG ===
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')  # Load API key from environment variable
DEEPSEEK_API_URL = "https://api.together.xyz/v1/chat/completions"  # Replace with actual DeepSeek endpoint
samplerate = 16000
model = whisper.load_model("base")
engine = pyttsx3.init()

def speak(text):
    print(f"🗣️ Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def record_audio(duration=5):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    return audio.flatten()

def transcribe(audio):
    audio_int16 = (audio * 32767).astype(np.int16)
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            scipy.io.wavfile.write(tmp.name, samplerate, audio_int16)
            result = model.transcribe(tmp.name)
        return result["text"]
    finally:
        # Ensure the temporary file is deleted even if an error occurs
        os.remove(tmp.name)

def respond_to(text):
    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",  # Correct model from API response
        "prompt": text,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        # Extract content from the 'choices' array and return the assistant's response
        return data["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"



def main():
    speak("Hi, I am Moomin! Ready to assist.")
    while True:
        try:
            audio = record_audio()
            text = transcribe(audio)
            print(f"📝 You said: {text}")
            if not text.strip():
                speak("I didn't catch that.")
                continue
            response = respond_to(text)
            speak(response)
            if "goodbye" in response.lower():
                break
        except KeyboardInterrupt:
            print("\n🎤 Stopped by user.")
            speak("Goodbye!")
            break
        except Exception as e:
            print("❌ Error:", e)
            speak("Something went wrong.")

if __name__ == "__main__":
    main()
