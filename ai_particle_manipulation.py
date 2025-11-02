import cv2
import numpy as np
import math
import random
import time
import threading
from collections import deque
import argparse
import json
import os

# Optional imports for audio and hand tracking
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    HAVE_SD = False

try:
    import mediapipe as mp
    HAVE_MP = True
except Exception:
    HAVE_MP = False

try:
    import pyaudio as pa
    HAVE_PA = True
except Exception:
    HAVE_PA = False


class AudioLevel:
    def __init__(self, samplerate=44100, blocksize=1024):
        self.level = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._stream = None
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._history = deque(maxlen=20)
        self._pa = None
        self._pa_stream = None

    def start(self):
        if self._running:
            return True

        if 'sd' in globals() and HAVE_SD:
            def callback(indata, frames, time_info, status):
                if status:
                    pass
                if indata is None or len(indata) == 0:
                    rms = 0.0
                else:
                    x = indata.astype(np.float32)
                    rms = float(np.sqrt(np.mean(np.square(x))))
                self._history.append(rms)
                smoothed = float(np.mean(self._history)) if self._history else 0.0
                with self._lock:
                    self.level = smoothed
            try:
                self._stream = sd.InputStream(callback=callback, channels=1, samplerate=self._samplerate, blocksize=self._blocksize)
                self._stream.start()
                self._running = True
                return True
            except Exception:
                self._stream = None
                self._running = False

        if 'pa' in globals() and HAVE_PA and not self._running:
            try:
                self._pa = pa.PyAudio()
                fmt = pa.paInt16
                ch = 1
                def pa_callback(in_data, frame_count, time_info, status_flags):
                    if in_data is None:
                        rms = 0.0
                    else:
                        data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                        rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
                    self._history.append(rms)
                    smoothed = float(np.mean(self._history)) if self._history else 0.0
                    with self._lock:
                        self.level = smoothed
                    return (None, pa.paContinue)
                self._pa_stream = self._pa.open(format=fmt, channels=ch, rate=self._samplerate, input=True, frames_per_buffer=self._blocksize, stream_callback=pa_callback)
                self._pa_stream.start_stream()
                self._running = True
                return True
            except Exception:
                try:
                    if self._pa_stream is not None:
                        if self._pa_stream.is_active():
                            self._pa_stream.stop_stream()
                        self._pa_stream.close()
                except Exception:
                    pass
                try:
                    if self._pa is not None:
                        self._pa.terminate()
                except Exception:
                    pass
                self._pa_stream = None
                self._pa = None
                self._running = False
        return False

    def get_level(self):
        with self._lock:
            return float(self.level)

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa_stream is not None:
            try:
                if self._pa_stream.is_active():
                    self._pa_stream.stop_stream()
                self._pa_stream.close()
            except Exception:
                pass
            self._pa_stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        self._running = False


class HandTracker:
    def __init__(self):
        self.enabled = HAVE_MP
        if not self.enabled:
            self._hands = None
            return
        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
            model_complexity=1,
        )
        self._last_open_val = {}

    def process(self, frame_bgr):
        if not self.enabled or self._hands is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            self._last_open_val = {}
            return None
        h, w, _ = frame_bgr.shape
        out = []
        hands_lm = results.multi_hand_landmarks
        hands_hd = getattr(results, 'multi_handedness', None)
        for idx, hand_landmarks in enumerate(hands_lm):
            xs, ys = [], []
            for lm in hand_landmarks.landmark:
                xs.append(lm.x * w)
                ys.append(lm.y * h)
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            tip_ids = [4, 8, 12, 16, 20]
            mcp_ids = [1, 5, 9, 13, 17]
            pts = [(hand_landmarks.landmark[i], hand_landmarks.landmark[j]) for i, j in zip(tip_ids, mcp_ids)]
            dists = []
            for tip, base in pts:
                dx = (tip.x - base.x) * w
                dy = (tip.y - base.y) * h
                dists.append(math.hypot(dx, dy))
            wrist = hand_landmarks.landmark[0]
            mid_mcp = hand_landmarks.landmark[9]
            scale = math.hypot((wrist.x - mid_mcp.x) * w, (wrist.y - mid_mcp.y) * h) + 1e-5
            openness = float(np.clip(np.mean(dists) / (1.6 * scale), 0.0, 1.0))
            label = None
            score = None
            if hands_hd and idx < len(hands_hd):
                try:
                    label = hands_hd[idx].classification[0].label
                    score = float(hands_hd[idx].classification[0].score)
                except Exception:
                    label = None
                    score = None
            key = label or f'H{idx}'
            prev_open = self._last_open_val.get(key)
            self._last_open_val[key] = openness
            out.append({
                'center': (cx, cy),
                'openness': openness,
                'was_open': None if prev_open is None else prev_open > 0.45,
                'is_open': openness > 0.45,
                'label': label,
                'score': score,
                'points': pts_px
            })
        return out if out else None

    def close(self):
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
            self._hands = None


class ParticleSystem:
    def __init__(self, n, width, height, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.width = width
        self.height = height
        self.n = n
        # State
        self.pos = np.column_stack([
            np.random.uniform(0, width, size=n).astype(np.float32),
            np.random.uniform(0, height, size=n).astype(np.float32)
        ])
        angle = np.random.uniform(0, 2*np.pi, size=n).astype(np.float32)
        speed = np.random.uniform(0.2, 1.2, size=n).astype(np.float32)
        self.vel = np.column_stack([np.cos(angle)*speed, np.sin(angle)*speed]).astype(np.float32)
        # Colors (neon-like: blues, purples, pinks)
        base_colors = np.array([
            [255, 50, 200],  # pink
            [255, 120, 20],  # purple-ish (BGR)
            [240, 160, 40],  # violet
            [220, 220, 20],  # magenta
            [200, 255, 40],  # blue
        ], dtype=np.float32)
        self.color = base_colors[np.random.randint(0, len(base_colors), size=n)]
        self.brightness = np.random.uniform(0.6, 1.0, size=(n, 1)).astype(np.float32)
        # For trail effect
        self.trail = np.zeros((height, width, 3), dtype=np.float32)
        self.last_burst_time = 0.0
        self._palette_index = 0

    def _apply_bounds(self):
        # Wrap around for a seamless effect
        self.pos[:, 0] = np.mod(self.pos[:, 0], self.width)
        self.pos[:, 1] = np.mod(self.pos[:, 1], self.height)

    def update(self, dt, hand_info, audio_level):
        # Base damping
        damping = 0.98
        self.vel *= damping

        # Gentle noise to keep motion alive; scaled by audio level
        jitter_amp = 25.0 * (0.2 + 2.5 * np.clip(audio_level*8.0, 0.0, 1.0))
        jitter = (np.random.uniform(-1, 1, size=self.vel.shape).astype(np.float32)) * (jitter_amp * dt)
        self.vel += jitter

        # Hand forces
        if hand_info is not None:
            hands = hand_info if isinstance(hand_info, list) else [hand_info]
            for hinfo in hands:
                hx, hy = hinfo['center']
                dx = hx - self.pos[:, 0]
                dy = hy - self.pos[:, 1]
                dist2 = dx*dx + dy*dy + 1e-4
                dist = np.sqrt(dist2)
                dirx = dx / dist
                diry = dy / dist
                is_open = hinfo.get('is_open', True)
                strength = (3000.0 / dist2)
                strength = np.clip(strength, 0.0, 3.0)
                label = hinfo.get('label')
                if label == 'Left':
                    self.vel[:, 0] += dirx * strength * 1.2
                    self.vel[:, 1] += diry * strength * 1.2
                    self.brightness[:, 0] = np.clip(self.brightness[:, 0] + (1.0/(1.0+dist))*0.02, 0.4, 1.5)
                elif label == 'Right':
                    self.vel[:, 0] -= dirx * strength * 1.35
                    self.vel[:, 1] -= diry * strength * 1.35
                    self.brightness[:, 0] = np.clip(self.brightness[:, 0] + (1.0/(1.0+dist))*0.03, 0.4, 1.7)
                else:
                    if is_open:
                        self.vel[:, 0] += dirx * strength * 1.2
                        self.vel[:, 1] += diry * strength * 1.2
                        self.brightness[:, 0] = np.clip(self.brightness[:, 0] + (1.0/(1.0+dist))*0.02, 0.4, 1.5)
                    else:
                        self.vel[:, 0] -= dirx * strength * 1.35
                        self.vel[:, 1] -= diry * strength * 1.35
                        self.brightness[:, 0] = np.clip(self.brightness[:, 0] + (1.0/(1.0+dist))*0.03, 0.4, 1.7)
                was_open = hinfo.get('was_open')
                if (label == 'Right' and was_open is True and is_open is False) or (label is None and was_open is True and is_open is False):
                    now = time.time()
                    if now - self.last_burst_time > 0.2:
                        angles = np.random.uniform(0, 2*np.pi, size=self.n).astype(np.float32)
                        burst_speed = np.random.uniform(100.0, 260.0, size=self.n).astype(np.float32)
                        self.vel[:, 0] += np.cos(angles) * burst_speed
                        self.vel[:, 1] += np.sin(angles) * burst_speed
                        self.last_burst_time = now

        # Audio also modulates brightness
        self.brightness[:, 0] = np.clip(self.brightness[:, 0] * (0.995 + np.clip(audio_level*0.2, 0.0, 0.2)), 0.4, 2.0)

        # Integrate
        self.pos += self.vel * dt
        self._apply_bounds()

    def set_palette(self, idx):
        palettes = [
            np.array([
                [255, 50, 200],
                [255, 120, 20],
                [240, 160, 40],
                [220, 220, 20],
                [200, 255, 40],
            ], dtype=np.float32),
            np.array([
                [255, 80, 80],
                [255, 200, 80],
                [200, 255, 200],
                [140, 220, 255],
                [220, 140, 255],
            ], dtype=np.float32),
            np.array([
                [255, 70, 160],
                [255, 100, 60],
                [230, 200, 60],
                [200, 220, 120],
                [160, 255, 200],
            ], dtype=np.float32),
        ]
        self._palette_index = idx % len(palettes)
        palette = palettes[self._palette_index]
        self.color = palette[np.random.randint(0, len(palette), size=self.n)]

    def cycle_palette(self):
        self.set_palette(self._palette_index + 1)

    def change_count(self, new_n):
        new_n = int(max(10, new_n))
        if new_n == self.n:
            return
        if new_n > self.n:
            add = new_n - self.n
            add_pos = np.column_stack([
                np.random.uniform(0, self.width, size=add).astype(np.float32),
                np.random.uniform(0, self.height, size=add).astype(np.float32)
            ])
            angles = np.random.uniform(0, 2*np.pi, size=add).astype(np.float32)
            speeds = np.random.uniform(0.2, 1.2, size=add).astype(np.float32)
            add_vel = np.column_stack([np.cos(angles)*speeds, np.sin(angles)*speeds]).astype(np.float32)
            add_bright = np.random.uniform(0.6, 1.0, size=(add, 1)).astype(np.float32)
            self.pos = np.vstack([self.pos, add_pos])
            self.vel = np.vstack([self.vel, add_vel])
            self.brightness = np.vstack([self.brightness, add_bright])
            self.n = new_n
            self.set_palette(self._palette_index)
        else:
            keep = new_n
            self.pos = self.pos[:keep]
            self.vel = self.vel[:keep]
            self.brightness = self.brightness[:keep]
            self.color = self.color[:keep]
            self.n = new_n

    def render(self, frame_bgr, alpha_decay=0.85, particle_size=2, blur_sigma=3.0):
        h, w, _ = frame_bgr.shape
        # Decay trails
        self.trail *= alpha_decay
        # Draw particles to trail buffer additively
        # Compute integer positions
        ix = np.clip(self.pos[:, 0].astype(np.int32), 0, w - 1)
        iy = np.clip(self.pos[:, 1].astype(np.int32), 0, h - 1)
        # Draw small discs by stamping a kernel
        radius = particle_size
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = (x*x + y*y <= radius*radius).astype(np.float32)
        # Precompute color per particle with brightness
        colors = (self.color * self.brightness).astype(np.float32)
        # Stamp particles (vectorized-ish)
        for (px, py), col in zip(zip(ix, iy), colors):
            x0 = max(px - radius, 0)
            x1 = min(px + radius + 1, w)
            y0 = max(py - radius, 0)
            y1 = min(py + radius + 1, h)
            kx0 = x0 - (px - radius)
            ky0 = y0 - (py - radius)
            kx1 = kx0 + (x1 - x0)
            ky1 = ky0 + (y1 - y0)
            k = mask[ky0:ky1, kx0:kx1][..., None]
            self.trail[y0:y1, x0:x1] += k * (col / 255.0) * 1.4
        # Clip trails
        np.clip(self.trail, 0.0, 1.5, out=self.trail)
        # Glow bloom: blur trail and add
        blur = cv2.GaussianBlur(self.trail, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        out = np.clip(self.trail * 0.6 + blur * 0.8, 0.0, 1.7)
        # Blend over frame
        blended = cv2.addWeighted(frame_bgr.astype(np.float32)/255.0, 0.55, out, 0.95, 0.0)
        return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def draw_gradient_background(h, w):
    # Dark radial gradient background
    cx, cy = w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    d_norm = d / (0.75 * max(w, h))
    d_norm = np.clip(d_norm, 0.0, 1.0)
    # Colors in BGR: deep blue to purple
    inner = np.array([20, 10, 40], dtype=np.float32)
    outer = np.array([4, 2, 8], dtype=np.float32)
    bg = (inner*(1.0 - d_norm[..., None]) + outer*(d_norm[..., None]))
    return bg.astype(np.uint8)


class Settings:
    def __init__(self):
        # Default settings
        self.camera_index = 0
        self.width = 1280
        self.height = 720
        self.particles = 400
        self.blur = 3.0
        self.alpha = 0.86
        self.particle_size = 2
        self.show_gradient = True
        self.show_webcam = True
        self.palette = 0
        self.mirror = True
        self.show_landmarks = False
        self.show_settings = False
        self.settings_file = 'particle_settings.json'
        
        # Load saved settings if they exist
        self.load()
    
    def to_dict(self):
        """Convert settings to dictionary for saving"""
        return {
            'camera_index': self.camera_index,
            'width': self.width,
            'height': self.height,
            'particles': self.particles,
            'blur': self.blur,
            'alpha': self.alpha,
            'particle_size': self.particle_size,
            'show_gradient': self.show_gradient,
            'show_webcam': self.show_webcam,
            'palette': self.palette,
            'mirror': self.mirror
        }
    
    def from_dict(self, data):
        """Load settings from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self):
        """Save settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")
    
    def load(self):
        """Load settings from file"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    self.from_dict(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load settings: {e}")


def draw_settings_panel(frame, settings, fps, hand_detected, audio_level):
    """Draw the settings panel on the frame"""
    if not settings.show_settings:
        return frame
    
    h, w = frame.shape[:2]
    panel_w = min(400, w // 3)
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    
    # Panel background with some transparency
    panel_bg = np.array([20, 20, 30], dtype=np.uint8)
    panel[:] = panel_bg
    
    # Add title and status
    cv2.putText(panel, "SETTINGS", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw settings controls (simplified for now)
    y_offset = 60
    settings_list = [
        (f"Particles: {settings.particles}", 'p'),
        (f"Blur: {settings.blur:.1f}", 'b'),
        (f"Alpha: {settings.alpha:.2f}", 'a'),
        (f"Size: {settings.particle_size}", 's'),
        (f"Gradient: {'ON' if settings.show_gradient else 'OFF'}", 'g'),
        (f"Webcam: {'ON' if settings.show_webcam else 'OFF'}", 'w'),
        (f"Palette: {settings.palette + 1}", 'c'),
        (f"Mirror: {'ON' if settings.mirror else 'OFF'}", 'm'),
        (f"Landmarks: {'ON' if settings.show_landmarks else 'OFF'}", 'h'),
        ("", ""),
        ("[S] Toggle Settings", ""),
        ("[ESC] Exit", "")
    ]
    
    for text, key in settings_list:
        if text:
            cv2.putText(panel, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
            if key:
                cv2.putText(panel, f"[{key.upper()}]", (panel_w - 40, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
        y_offset += 30
    
    # Add system status
    y_offset += 10
    cv2.line(panel, (10, y_offset), (panel_w-10, y_offset), (100, 100, 100), 1)
    y_offset += 20
    
    status = [
        f"FPS: {fps:.1f}",
        f"Hands: {'DETECTED' if hand_detected else 'none'}",
        f"Audio: {audio_level*100:.0f}%"
    ]
    
    for text in status:
        cv2.putText(panel, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)
        y_offset += 25
    
    # Blend panel with frame
    panel_roi = frame[0:h, 0:panel_w]
    frame[0:h, 0:panel_w] = cv2.addWeighted(panel, 0.8, panel_roi, 0.2, 0)
    return frame


def main():
    title = 'AI Particle Manipulation'
    parser = argparse.ArgumentParser(prog=title)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--particles', type=int, default=400)
    parser.add_argument('--blur', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=0.86)
    parser.add_argument('--size', type=int, default=2)
    parser.add_argument('--no-gradient', action='store_true')
    parser.add_argument('--no-webcam', action='store_true')
    parser.add_argument('--palette', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=0.0)
    parser.add_argument('--mirror', action='store_true')
    args = parser.parse_args()

    def open_any_camera(pref_index, w_hint=None, h_hint=None):
        order = []
        seen = set()
        for i in [pref_index, 0, 1, 2, 3, 4, 5]:
            ii = int(i)
            if ii not in seen:
                order.append(ii)
                seen.add(ii)
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]
        for idx in order:
            for be in backends:
                try:
                    capx = cv2.VideoCapture(idx, be)
                    if not capx.isOpened():
                        capx = cv2.VideoCapture(idx)
                    if w_hint:
                        try:
                            capx.set(cv2.CAP_PROP_FRAME_WIDTH, int(w_hint))
                        except Exception:
                            pass
                    if h_hint:
                        try:
                            capx.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h_hint))
                        except Exception:
                            pass
                    ok, fr = capx.read()
                    if ok and fr is not None and fr.size > 0:
                        return capx, fr, idx
                    try:
                        capx.release()
                    except Exception:
                        pass
                except Exception:
                    pass
        fw = int(w_hint) if w_hint else 1280
        fh = int(h_hint) if h_hint else 720
        return None, np.zeros((fh, fw, 3), dtype=np.uint8), int(pref_index)

    cap, frame, cam_index = open_any_camera(args.camera, args.width, args.height)
    h, w = frame.shape[:2]

    # Initialize systems
    settings = Settings()
    ps = ParticleSystem(
        n=settings.particles,
        width=w,
        height=h,
        seed=42
    )
    ps.set_palette(settings.palette)
    hands = HandTracker()
    audio = AudioLevel()
    audio.start()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, w, h)

    fps_hist = deque(maxlen=30)
    last = time.time()
    settings.show_gradient = not args.no_gradient
    settings.show_webcam = not args.no_webcam
    settings.blur = float(args.blur)
    settings.alpha = float(args.alpha)
    draw_hands = False
    last_tune = time.time()
    start_time = time.time()
    cam_index = int(cam_index)
    cam_fail_count = 0
    mirror_mode = bool(args.mirror)
    hands_fail_count = 0

    fps_frames = 0
    fps_timer = time.time()

    while True:
        ret, frame = (cap.read() if cap is not None else (False, None))
        if not ret or frame is None:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cam_fail_count += 1
        else:
            # Resize to initial size for consistency
            frame = cv2.resize(frame, (w, h))
            cam_fail_count = 0
        if mirror_mode:
            frame = cv2.flip(frame, 1)

        webcam_bg = (frame * 0.18).astype(np.uint8)
        gradient_bg = draw_gradient_background(h, w)
        if settings.show_gradient and settings.show_webcam:
            base = cv2.addWeighted(gradient_bg, 0.75, webcam_bg, 0.25, 0)
        elif settings.show_gradient and not settings.show_webcam:
            base = gradient_bg
        elif not settings.show_gradient and settings.show_webcam:
            base = webcam_bg
        else:
            base = np.zeros_like(frame)

        # Hand tracking
        hand_info = hands.process(frame) if HAVE_MP else None
        if hand_info is None:
            hands_fail_count += 1
        else:
            hands_fail_count = 0

        # Update particles
        now = time.time()
        dt = max(0.001, now - last)
        last = now
        audio_level = audio.get_level() if getattr(audio, '_running', False) else 0.0
        ps.update(dt, hand_info, audio_level)
        
        # Update FPS counter
        fps_frames += 1
        current_time = time.time()
        if current_time - fps_timer >= 1.0:
            fps = fps_frames / (current_time - fps_timer)
            fps_timer = current_time
            fps_frames = 0

        # Render particles
        display = ps.render(base, alpha_decay=settings.alpha, 
                          particle_size=settings.particle_size, 
                          blur_sigma=settings.blur)

        # Draw settings panel if enabled
        if settings.show_settings:
            display = draw_settings_panel(display, settings, fps, hand_info is not None, audio_level)

        # UI overlays
        # Title
        cv2.putText(display, title, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 150, 255), 2, cv2.LINE_AA)
        # FPS
        fps_hist.append(1.0 / dt)
        fps = float(np.mean(fps_hist)) if fps_hist else 0.0
        cv2.putText(display, f'FPS: {fps:5.1f}', (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        # Status indicators
        mic_txt = 'ON' if getattr(audio, '_running', False) else 'OFF'
        hand_txt = 'ON' if HAVE_MP and hand_info is not None else 'OFF'
        cam_txt = 'ON' if cam_fail_count == 0 else 'OFF'
        cv2.putText(display, f'Mic: {mic_txt}', (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f'Hands: {hand_txt}', (18, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f'Camera: {cam_txt}', (18, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 2, cv2.LINE_AA)

        if cam_fail_count > 15:
            cv2.putText(display, 'No camera frames. Press n/p to switch camera, or check permissions.', (18, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 2, cv2.LINE_AA)

        if hand_info is not None:
            infos = hand_info if isinstance(hand_info, list) else [hand_info]
            y0 = 200 if cam_fail_count > 15 else 170
            for idx, hinfo in enumerate(infos):
                label = hinfo.get('label') or 'Hand'
                if 'center' in hinfo and hinfo['center'] is not None:
                    cx, cy = map(int, hinfo['center'])
                    cv2.circle(display, (cx, cy), 16, (255, 200, 100), 2)
                state = 'OPEN' if hinfo.get('is_open') else 'CLOSED'
                cv2.putText(display, f'{label}: {state}', (18, y0 + idx*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 150), 2, cv2.LINE_AA)
            if draw_hands:
                try:
                    connections = list(mp.solutions.hands.HAND_CONNECTIONS) if HAVE_MP else []
                except Exception:
                    connections = []
                for hinfo in infos:
                    pts = hinfo.get('points') or []
                    for p in pts:
                        cv2.circle(display, tuple(p), 2, (80, 255, 200), -1)
                    for (i, j) in connections:
                        if i < len(pts) and j < len(pts):
                            cv2.line(display, tuple(pts[i]), tuple(pts[j]), (120, 220, 255), 1, cv2.LINE_AA)

        cv2.imshow(title, display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('x'):
            break
        if key == ord('+') or key == ord('='):
            settings.particles = min(10000, settings.particles + 100)
            ps.change_count(settings.particles)
            settings.save()
        if key == ord('-'):
            settings.particles = max(100, settings.particles - 100)
            ps.change_count(settings.particles)
            settings.save()
        if key == ord('c'):
            settings.palette = (settings.palette + 1) % 5  # Assuming 5 palettes
            ps.set_palette(settings.palette)
            settings.save()
        if key == ord('g'):
            settings.show_gradient = not settings.show_gradient
            settings.save()
        if key == ord('w'):
            settings.show_webcam = not settings.show_webcam
            settings.save()
        if key == ord('h'):
            draw_hands = not draw_hands
        if key == ord('n'):
            try:
                cam_index += 1
                new_cap, fr, cam_index = open_any_camera(cam_index, settings.width, settings.height)
                old = cap
                cap = new_cap
                if old is not None:
                    try:
                        old.release()
                    except Exception:
                        pass
            except Exception:
                pass
        if key == ord('p'):
            try:
                cam_index = max(0, cam_index - 1)
                new_cap, fr, cam_index = open_any_camera(cam_index, settings.width, settings.height)
                old = cap
                cap = new_cap
                if old is not None:
                    try:
                        old.release()
                    except Exception:
                        pass
            except Exception:
                pass

        if key == ord('r'):
            try:
                new_cap, fr, cam_index = open_any_camera(cam_index, settings.width, settings.height)
                old = cap
                cap = new_cap
                if old is not None:
                    try:
                        old.release()
                    except Exception:
                        pass
                cam_fail_count = 0
            except Exception:
                pass

        if time.time() - last_tune > 1.0:
            cur_fps = float(np.mean(fps_hist)) if fps_hist else 0.0
            if cur_fps < 30.0:
                settings.blur = max(1.4, settings.blur - 0.2)
                settings.alpha = max(0.80, settings.alpha - 0.01)
                if ps.n > 200:
                    ps.change_count(ps.n - 20)
            elif cur_fps > 45.0:
                settings.blur = min(4.0, settings.blur + 0.2)
                settings.alpha = min(0.90, settings.alpha + 0.005)
                if ps.n < 600:
                    ps.change_count(ps.n + 20)
            if cur_fps < 22.0 and settings.show_webcam:
                settings.show_webcam = False
            last_tune = time.time()

        if hands_fail_count > 120:
            mirror_mode = not mirror_mode
            hands_fail_count = 0

        if cam_fail_count > 60:
            try:
                cam_index += 1
                new_cap, fr, cam_index = open_any_camera(cam_index, args.width, args.height)
                old = cap
                cap = new_cap
                if old is not None:
                    try:
                        old.release()
                    except Exception:
                        pass
                cam_fail_count = 0
            except Exception:
                pass

        if args.timeout and (time.time() - start_time) > args.timeout:
            break

    try:
        audio.stop()
    except Exception:
        pass
    try:
        hands.close()
    except Exception:
        pass
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
