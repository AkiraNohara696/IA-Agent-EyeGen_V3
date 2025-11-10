# core/http_sender.py
from __future__ import annotations
import threading, queue, requests, cv2

class HttpSender:
    def __init__(self, url: str, timeout=2.0, max_q=2):
        self.url = url
        self.timeout = timeout
        self.q = queue.Queue(maxsize=max_q)  # “mailbox” pequeno para não acumular
        self.running = False
        self.th = None

    def start(self):
        self.running = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        sess = requests.Session()
        while self.running:
            try:
                frame = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            # JPEG encode
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            data = buf.tobytes()
            try:
                # envie como multipart ou octet-stream
                sess.post(self.url, files={"frame": ("frame.jpg", data, "image/jpeg")},
                          timeout=self.timeout)
            except Exception:
                pass

    def submit(self, frame):
        # drop oldest if cheio -> não acumula latência
        if self.q.full():
            try:
                self.q.get_nowait()
            except Exception:
                pass
        try:
            self.q.put_nowait(frame)
        except Exception:
            pass

    def stop(self):
        self.running = False
        try:
            if self.th:
                self.th.join(timeout=0.5)
        except Exception:
            pass
