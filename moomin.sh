#!/bin/bash

echo "🎤 Listening..."

# Step 1: Record audio (4 seconds)
ffmpeg -loglevel quiet -f alsa -i default -t 4 -ar 16000 -ac 1 /tmp/input.wav

# Step 2: Transcribe using Whisper
whisper /tmp/input.wav --model tiny --output_format txt --output_dir /tmp > /dev/null
TRANSCRIPT=$(cat /tmp/input.txt)

echo "📝 You said: $TRANSCRIPT"

# Step 3: Send transcript to GPT-4 (via Python)
RESPONSE=$(python3 /opt/muomin/ask_gpt.py "$TRANSCRIPT")

# Step 4: Notify + Speak
notify-send "Mo'min OS" "$RESPONSE"
espeak-ng "$RESPONSE"

