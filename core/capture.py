# core/capture.py
from __future__ import annotations
import cv2, threading, time, platform

def _open_capture(src, buffersize=1):
    is_win = platform.system().lower().startswith("win")
    # Se for webcam (int), use backends de câmera nativos (menor latência)
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if is_win else 0)
        # Tente um formato leve (MJPG) com FPS alto
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass
        return cap
    # Caso contrário (URL), mantenha FFMPEG e buffersize curto
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    except Exception:
        pass
    return cap

class FrameGrabber:
    def __init__(self, src, backend=None, buffersize=1):
        # backend é ignorado agora; escolhemos automaticamente
        self.cap = _open_capture(src, buffersize=buffersize)
        self.latest = None
        self.running = False
        self.lock = threading.Lock()
        self.th = None

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def start(self):
        if not self.is_opened():
            return False
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return True

    def _loop(self):
        # leitura contínua; sempre substitui pelo frame mais recente
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest = frame

    def get(self):
        with self.lock:
            return None if self.latest is None else self.latest.copy()

    def stop(self):
        self.running = False
        try:
            if self.th:
                self.th.join(timeout=0.5)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
