# AI Particle Manipulation

Real-time, interactive particle art controlled by your hands and microphone. Uses OpenCV for rendering, MediaPipe Hands for tracking, and NumPy for fast particle physics. Includes a settings panel and resilient camera handling.

## Features
- Hand tracking with MediaPipe Hands (open vs. closed fist)
- Attraction/repulsion forces near each hand; burst on fist close
- Neon particles with glow trails and bloom
- Sound-reactive brightness/vibrations (mic input)
- Webcam background blended with a dark radial gradient
- HUD with FPS/Mic/Hands/Camera status
- Robust camera scanning with hotkeys (n/p/r)
- Persistent settings saved to `particle_settings.json`

## Requirements
- Windows 10/11, Python 3.9–3.12
- Webcam and microphone
- No GPU required

## Installation
```bash
python -m venv .venv
.venv\Scripts\pip install -U pip
.venv\Scripts\pip install -r requirements.txt
```

If MediaPipe fails for your Python version, try upgrading pip first or install a matching wheel from https://pypi.org/project/mediapipe/.

If `sounddevice` cannot open the mic, make sure the input device is enabled under Windows Settings → Privacy → Microphone.

## Run
```bash
.venv\Scripts\python ai_particle_manipulation.py --mirror
```

## Hotkeys
- **s**: Toggle settings panel
- **h**: Toggle hand landmarks overlay
- **n/p**: Next/previous camera
- **r**: Rescan cameras
- **+ / -**: Increase/decrease particle count
- **g**: Toggle gradient background
- **w**: Toggle webcam overlay
- **c**: Cycle color palettes
- **x / Esc**: Quit

## Settings Panel
Press `s` to open. Adjust particle count, blur, alpha, palette, gradient/webcam visibility, mirror mode. Settings are persisted across runs in `particle_settings.json`.

## Troubleshooting
- **Camera OFF/black**: Press `n`/`p` to switch cameras or `r` to rescan. Ensure no other app uses the webcam. Allow Camera access in Windows privacy settings.
- **Hands OFF**: Improve lighting; move your hand closer; press `h` for landmarks.
- **Mic OFF**: Confirm mic permission and device is active; try installing `pyaudio` if `sounddevice` isn’t available.
- **Performance**: Reduce particles, increase blur a bit, or hide the webcam overlay.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Credits
- MediaPipe Hands (Google)
- OpenCV, NumPy

---
Maintained by peddapudisiva. Pull requests and issues welcome.
