# app.py
from __future__ import annotations

from core import config
config.apply_perf_env()

import os, cv2, torch, numpy as np, time
from time import perf_counter
from threading import Thread, Lock
from flask import Flask, Response  # >>> LIVE
from queue import Queue, Empty     # <<< TTS assíncrono

from core.capture import FrameGrabber
from core.http_sender import HttpSender
from ui.yolo_loader import load_model
from core.heuristics import center_offset, choose_direction, bin_distance_label
from core.smoothing import smooth_area, smooth_center, estimate_distance_bins_from_rel_area
from core.tracking import upsert_track, is_approaching
from ui.hud import HudStabilizer
from ui.talk_gate import TalkGate
from core.tts import init_tts, falar, iterate, end_loop
from llm.signature import init_llm, predict_line

# --------- Perf ----------
try:
    cv2.setNumThreads(1)
    cv2.setUseOptimized(True)  # ativa otimizações do OpenCV
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
try:
    torch.set_num_threads(config.OMP_THREADS)
    torch.set_num_interop_threads(max(1, config.OMP_THREADS // 2))
except Exception:
    pass

# >>> LIVE: buffer do preview JPEG (sempre o mais recente)
_live_lock = Lock()
_live_jpeg = None
_JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, int(os.getenv("ASSISTIVENAV_JPEG_QUALITY", "60"))]  # 50–70 = menos lag

def _set_live_frame(bgr):
    global _live_jpeg
    ok, jpg = cv2.imencode(".jpg", bgr, _JPEG_PARAMS)
    if ok:
        with _live_lock:
            _live_jpeg = jpg.tobytes()

# >>> LIVE: servidor Flask simples para MJPEG
_app = Flask(__name__)

@_app.route("/live")
def live():
    def gen():
        boundary = b"--frame\r\n"
        idle_sleep = 1/45.0  # throttle ~45fps de envio; evita busy-loop
        while True:
            with _live_lock:
                blob = _live_jpeg
            if blob is None:
                time.sleep(0.01)  # evita busy-wait e não bloqueia a UI
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + blob + b"\r\n"
            time.sleep(idle_sleep)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def _run_live_server():
    port = int(os.getenv("ASSISTIVENAV_LIVE_PORT", "8081"))
    _app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

def _resize_square(frame, size: int) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

# ======= TTS assíncrono (thread + fila) =======
class TTSWorker(Thread):
    def __init__(self, cooldown_s: float = 0.0):
        super().__init__(daemon=True)
        self.q = Queue()
        self._run = True
        self._last = ""
        self._cooldown_s = float(cooldown_s)
        self._last_ts = 0.0

    def enqueue(self, text: str):
        now = time.time()
        if text and (text != self._last or (now - self._last_ts) >= self._cooldown_s):
            self.q.put(text)
            self._last = text
            self._last_ts = now

    def stop(self):
        self._run = False
        self.q.put(None)  # desbloqueia .get()

    def run(self):
        while self._run:
            try:
                text = self.q.get(timeout=0.25)
            except Empty:
                continue
            if text is None:
                break
            try:
                # Chamada bloqueante acontece AQUI, fora do loop de vídeo
                falar(text)
            except Exception as e:
                print("⚠️ Erro no TTS:", e)

def main():
    # ======= janela e captura =======
    cv2.namedWindow("AssistiveNav DSPy + YOLO", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AssistiveNav DSPy + YOLO", max(360, config.DISPLAY_TARGET_W), 480)

    grab = FrameGrabber(config.DEFAULT_CAMERA_URL, buffersize=1)
    if not grab.is_opened() or not grab.start():
        print(f"❌ Não foi possível abrir a câmera: {config.DEFAULT_CAMERA_URL}")
        return

    # ======= LIVE MJPEG =======
    if os.getenv("ASSISTIVENAV_LIVE", "1") == "1":
        Thread(target=_run_live_server, daemon=True).start()
        print("🌐 Live MJPEG em http://127.0.0.1:8081/live")

    # ======= infra auxiliar =======
    sender = HttpSender(url=os.getenv("ASSISTIVENAV_POST_URL", "http://127.0.0.1:5000/frame"))
    sender.start()

    init_tts()
    try:
        init_llm()
    except Exception:
        pass

    # ======= modelo + warmup =======
    model, backend = load_model()      # "onnx" | "pt"
    imgsz = int(getattr(config, "YOLO_IMGSZ", 416))
    with torch.inference_mode():
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            warm_device = 0 if (backend == "pt" and torch.cuda.is_available()) else "cpu"
            model.predict(source=dummy, device=warm_device, imgsz=imgsz, verbose=False)
        except Exception:
            pass

    # ======= HUD/Talk =======
    hud = HudStabilizer(hold_frames=config.HUD_HOLD_FRAMES, fade_to_none=config.HUD_FADE_TO_NONE)
    talk_hud = TalkGate(min_hold_frames=config.HUD_TALK_MIN_HOLD, cooldown_s=config.TALK_COOLDOWN_S)
    talk_evt = TalkGate(min_hold_frames=config.TALK_EVT_MIN_HOLD, cooldown_s=config.TALK_COOLDOWN_S)

    # ======= TTS worker (paralelo) =======
    tts = TTSWorker(cooldown_s=float(os.getenv("ASSISTIVENAV_TTS_COOLDOWN", str(config.TALK_COOLDOWN_S))))
    tts.start()

    # ======= estado compartilhado (dets e texto) =======
    state_lock = Lock()
    dets = []
    hud_text = "Inicializando..."
    last_ts = 0.0

    def set_state(new_dets, new_text):
        nonlocal dets, hud_text, last_ts
        with state_lock:
            dets = new_dets
            hud_text = new_text or "Nenhum objeto"
            last_ts = perf_counter()

    def get_state(max_age_s=1.0):
        with state_lock:
            age = perf_counter() - last_ts
            fresh = (age <= max_age_s)
            return (dets if fresh else []), hud_text, last_ts

    # ======= worker assíncrono de inferência =======
    class AsyncInfer(Thread):
        def __init__(self, model, backend, imgsz, every_frames, budget_ms):
            super().__init__(daemon=True)
            self.model = model
            self.backend = backend
            self.imgsz = imgsz
            self.every = max(1, int(every_frames))
            self.budget_ms = int(budget_ms)
            self._lock = Lock()
            self._frame = None
            self._run = True
            self._idx = 0

        def submit(self, frame):
            # guarda somente o mais recente
            with self._lock:
                self._frame = frame

        def stop(self):
            self._run = False

        def run(self):
            while self._run:
                with self._lock:
                    f = self._frame
                if f is None:
                    time.sleep(0.001)
                    continue

                self._idx += 1
                if self._idx % self.every != 0:
                    time.sleep(0.0005)
                    continue

                infer_frame = _resize_square(f, self.imgsz)
                t0 = perf_counter()
                results = []
                try:
                    dev = "cpu" if self.backend == "onnx" else ("cuda:0" if torch.cuda.is_available() else "cpu")
                    with torch.inference_mode():
                        results = self.model.predict(
                            source=infer_frame,
                            device=dev,
                            imgsz=self.imgsz,
                            conf=getattr(config, "YOLO_CONF", 0.35),
                            iou=getattr(config, "YOLO_IOU", 0.50),
                            max_det=30,
                            agnostic_nms=False,
                            augment=False,
                            verbose=False,
                        )
                except Exception as e:
                    print("⚠️ Erro na inferência:", e)
                    set_state([], "Nenhum objeto")
                    time.sleep(0.001)
                    continue

                ih, iw = infer_frame.shape[:2]
                area_total = float(iw * ih)

                # coleta detecções
                new_dets = []
                for r in results or []:
                    boxes = getattr(r, "boxes", [])
                    names = getattr(r, "names", {}) or {}
                    for b in boxes:
                        try:
                            cls_id = int(b.cls[0].item())
                            label = names.get(cls_id, str(cls_id))
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            conf = float(b.conf[0].item())
                            new_dets.append({"label": label, "xyxy": (x1, y1, x2, y2), "conf": conf})
                        except Exception:
                            continue

                # alvo + texto (leve)
                target, best_score = None, 1e9
                for d in new_dets:
                    x1, y1, x2, y2 = d["xyxy"]
                    c_off = center_offset(d["xyxy"], iw)
                    c_score = abs(c_off)
                    a = max(1.0, (x2 - x1)) * max(1.0, (y2 - y1))
                    inv_a = 1.0 - min(1.0, a / (area_total + 1e-6))
                    score = c_score + inv_a
                    if score < best_score:
                        best_score, target = score, d

                text = "Nenhum objeto"
                if target:
                    x1, y1, x2, y2 = target["xyxy"]
                    w_box = max(1.0, x2 - x1); h_box = max(1.0, y2 - y1)
                    box_area = w_box * h_box
                    area_s = smooth_area(target["label"], box_area)
                    rel_area = area_s / (area_total + 1e-6)
                    c_off_raw = center_offset(target["xyxy"], iw)
                    c_off = smooth_center(target["label"], c_off_raw)
                    dist_m = estimate_distance_bins_from_rel_area(rel_area)
                    direction = choose_direction(c_off, dist_m)
                    text = f"{direction}; {bin_distance_label(dist_m)}; {target['label']} à frente"

                set_state(new_dets, text)

                # autotune de cadência (mais responsivo; limites seguros)
                dt_ms = (perf_counter() - t0) * 1000.0
                if dt_ms > self.budget_ms and self.every < 8:
                    self.every += 1
                elif dt_ms < self.budget_ms * 0.5 and self.every > 2:
                    self.every -= 1

    worker = AsyncInfer(
        model=model,
        backend=backend,
        imgsz=imgsz,
        every_frames=int(os.getenv("PROC_EVERY", str(getattr(config, "PROC_EVERY", 6)))),  # default mais agressivo
        budget_ms=int(os.getenv("INFER_MAX_MS", str(getattr(config, "INFER_MAX_MS", 90)))),
    )
    worker.start()

    # ======= loop principal (preview first) =======
    OVERLAY_INTERVAL = float(os.getenv("ASSISTIVENAV_OVERLAY_INTERVAL", "0.5"))  # pulso/piscada
    OVERLAY_TTL = float(os.getenv("ASSISTIVENAV_OVERLAY_TTL", "0.20"))           # TTL curto = “piscada” visível
    last_overlay_t = 0.0
    overlay_cache = None  # np.ndarray com o desenho das boxes/labels
    overlay_cache_ts = 0.0

    # >>> Controle de fala (desacoplado do HUD)
    SPEAK_COOLDOWN = float(os.getenv("ASSISTIVENAV_TTS_COOLDOWN", str(getattr(config, "TALK_COOLDOWN_S", 1.0))))
    last_spoken_text = ""
    last_spoken_ts = 0.0

    try:
        while True:
            f = grab.get()
            if f is None:
                iterate()
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
                continue

            # manda frame mais novo pro worker (não bloqueia)
            worker.submit(f)

            # base do display (preview fluido)
            disp = _resize_square(f, imgsz)

            # pega últimas detecções/texto (não bloqueia)
            cur_dets, cur_text, det_ts = get_state(max_age_s=1.0)

            # (1) atualiza cache do overlay somente a cada OVERLAY_INTERVAL
            now = time.time()
            if (now - last_overlay_t) >= OVERLAY_INTERVAL:
                last_overlay_t = now
                # (2) respeita TTL das detecções: se velho, não atualiza cache
                det_age = perf_counter() - det_ts
                if cur_dets and det_age <= OVERLAY_TTL:
                    box_layer = np.zeros_like(disp)
                    for d in cur_dets:
                        x1, y1, x2, y2 = map(int, d["xyxy"])
                        cv2.rectangle(box_layer, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # rótulos com menor frequência (já no mesmo intervalo)
                    for d in cur_dets:
                        x1, y1, _, _ = map(int, d["xyxy"])
                        cv2.putText(box_layer, f"{d['label']} {d['conf']:.2f}",
                                    (x1, max(0, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    overlay_cache = box_layer
                    overlay_cache_ts = perf_counter()
                else:
                    # sem detecções recentes ou muito velhas -> limpa cache
                    overlay_cache = None
                    overlay_cache_ts = 0.0

            # (3) aplica overlay cache somente se cache for recente (TTL)
            if overlay_cache is not None and (perf_counter() - overlay_cache_ts) <= OVERLAY_TTL:
                if overlay_cache.shape == disp.shape:
                    disp = cv2.addWeighted(disp, 1.0, overlay_cache, 1.0, 0.0)
            # else: sem overlay (frame “respira”), mas preview segue fluido

            # HUD leve (estabilização só visual)
            stable_text = hud.update(cur_text)
            band = disp.copy()
            cv2.rectangle(band, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
            disp = cv2.addWeighted(band, 0.35, disp, 0.65, 0.0)
            cv2.putText(disp, stable_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)

            # publica no MJPEG e mostra
            _set_live_frame(disp)
            cv2.imshow("AssistiveNav DSPy + YOLO", disp)

            # TTS housekeeping (não bloqueante) — dispara com base no cur_text
            iterate()  # se seu core.tts usa, mantém (barato)
            now = time.time()
            if cur_text and cur_text != "Nenhum objeto":
                if (cur_text != last_spoken_text) and ((now - last_spoken_ts) >= SPEAK_COOLDOWN):
                    if talk_evt.update_and_should_speak(cur_text):
                        tts.enqueue(cur_text)  # fala em paralelo (sem travar frame)
                        last_spoken_text = cur_text
                        last_spoken_ts = now

            # nunca bloqueie aqui
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                break

    finally:
        try:
            worker.stop()
        except:
            pass
        try:
            sender.stop()
        except:
            pass
        try:
            grab.stop()
        except:
            pass
        try:
            tts.stop()     # encerra o worker de TTS
        except:
            pass
        cv2.destroyAllWindows()
        try:
            end_loop()
        except:
            pass


if __name__ == "__main__":
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
    except Exception:
        pass
    main()
