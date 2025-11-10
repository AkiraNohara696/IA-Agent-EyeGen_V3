# core/infer_worker.py
from __future__ import annotations
import threading, time
from collections import deque
import numpy as np

class InferWorker:
    """
    - Recebe frames (sempre substitui pelo mais recente)
    - Roda YOLO no background
    - Exponibiliza ultimo resultado pronto (detections + metadados)
    """
    def __init__(self, model, device: str, predict_kwargs: dict, max_queue=2):
        self.model = model
        self.device = device
        self.kw = predict_kwargs
        self._in_q = deque(maxlen=max_queue)   # buffer de entrada (último frame vence)
        self._out = None                       # último resultado pronto
        self._lock_out = threading.Lock()
        self._stop = False
        self._th = None

    def start(self):
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return True

    def stop(self):
        self._stop = True
        try:
            if self._th:
                self._th.join(timeout=0.5)
        except Exception:
            pass

    def submit(self, frame: np.ndarray):
        # apenas substitui; não bloqueia UI
        self._in_q.append(frame)

    def get_last(self):
        with self._lock_out:
            return self._out

    def _loop(self):
        import torch
        from time import perf_counter
        with torch.inference_mode():
            while not self._stop:
                if not self._in_q:
                    time.sleep(0.002)
                    continue
                frame = self._in_q.pop()  # pegue o mais novo
                t0 = perf_counter()
                try:
                    res = self.model.predict(source=frame, device=self.device, **self.kw)
                except Exception as e:
                    res, t_infer = [], 0.0
                else:
                    t_infer = (perf_counter() - t0) * 1000.0
                with self._lock_out:
                    self._out = (res, t_infer, frame.shape[:2])
