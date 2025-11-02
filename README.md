# AI Particle Manipulation

Real-time, interactive particle simulation controlled by hand gestures and microphone audio. Built with OpenCV, MediaPipe, NumPy, and SoundDevice.

## Features
- Hand tracking with MediaPipe Hands (open vs closed fist)
- Attraction/repulsion forces around the hand
- Particle bursts on fist close
- 400 neon particles with glow trails and bloom
- Sound-reactive brightness/vibration using microphone input
- Webcam background blended with a dark gradient
- On-screen title and FPS counter

## Requirements
- Windows 10/11, Python 3.9–3.11
- A working webcam and microphone
- GPU not required

## Install
```bash
python -m venv .venv
.venv\Scripts\pip install -U pip
.venv\Scripts\pip install -r requirements.txt
```

If MediaPipe wheel fails for your Python version, try upgrading pip first, or install the matching version from https://pypi.org/project/mediapipe/.

If `sounddevice` cannot open the mic, ensure your input device is enabled in Windows privacy settings.

## Run
```bash
.venv\Scripts\python ai_particle_manipulation.py
```

Keys: `Esc` or `q` to quit

## Tips
- If FPS is low, reduce particle count in `ParticleSystem(n=400, ...)` inside `ai_particle_manipulation.py`.
- You can also increase `alpha_decay` or reduce `particle_size` in `particles.render(...)`.
- Mic level is shown as ON/OFF. If OFF, audio effects will default to minimal.

## Troubleshooting
- Camera not opening: another app may be using it; close others or switch device index in `cv2.VideoCapture(0, ...)`.
- Black window: allow camera/microphone in Windows Settings → Privacy.
- No hand detection: ensure good lighting and visible hand in frame.
- Crackling audio/latency: reduce `blocksize` in `AudioLevel(samplerate=44100, blocksize=1024)`.
