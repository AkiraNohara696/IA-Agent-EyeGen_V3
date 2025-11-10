# core/config.py
from __future__ import annotations
import os

# ---------- Perf (bem econômico) ----------
OMP_THREADS = int(os.getenv("OMP_NUM_THREADS", "1"))
MKL_THREADS = int(os.getenv("MKL_NUM_THREADS", "1"))
KMP_DUPLICATE_LIB_OK = os.getenv("KMP_DUPLICATE_LIB_OK", "TRUE")

def apply_perf_env() -> None:
    """
    Chame ANTES de importar torch/numpy/ultralytics.
    """
    os.environ["OMP_NUM_THREADS"] = str(OMP_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(MKL_THREADS)
    os.environ["KMP_DUPLICATE_LIB_OK"] = KMP_DUPLICATE_LIB_OK

# ---------- LLM ----------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
USE_LLM = bool(OPENAI_KEY)
LLM_MODEL = os.getenv("ASSISTIVENAV_LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = int(os.getenv("ASSISTIVENAV_LLM_MAX_TOKENS", "40"))
LLM_TEMPERATURE = float(os.getenv("ASSISTIVENAV_LLM_TEMPERATURE", "0.2"))

# ---------- Vídeo ----------
DEFAULT_CAMERA_URL = os.getenv("DEFAULT_CAMERA_URL", "http://192.168.15.110:8080/video")
DISPLAY_TARGET_W = 416  # já combina com imgsz=416
# 0 = webcam notebook
#DEFAULT_CAMERA_URL = int(os.getenv("ASSISTIVENAV_CAMERA_URL", "0"))
#DISPLAY_TARGET_W   = int(os.getenv("ASSISTIVENAV_DISPLAY_W", "480"))

# A cada quantos frames roda a YOLO (maior = mais fluido, menos responsivo)
PROC_EVERY = int(os.getenv("ASSISTIVENAV_PROC_EVERY", "14"))

# ---------- YOLO ----------
# ATENÇÃO: pelo seu erro, o ONNX foi exportado com input fixo 640.
# Se re-exportar para 416 ou com dynamic axes, ajuste YOLO_IMGSZ/INFER_TARGET_W.
PREFER_ONNX     = True
YOLO_ONNX_PATH  = os.getenv("ASSISTIVENAV_YOLO_ONNX", "yolov8n.onnx")
YOLO_PT_PATH    = os.getenv("ASSISTIVENAV_YOLO_PT",   "yolov8n.pt")

YOLO_IMGSZ = int(os.getenv("ASSISTIVENAV_YOLO_IMGSZ", "416"))  # deve bater com o ONNX
YOLO_CONF  = float(os.getenv("ASSISTIVENAV_YOLO_CONF", "0.35"))
YOLO_IOU   = float(os.getenv("ASSISTIVENAV_YOLO_IOU",  "0.50"))

# Redimensiona ANTES da YOLO para o mesmo tamanho do ONNX
INFER_TARGET_W = int(os.getenv("ASSISTIVENAV_INFER_W", str(YOLO_IMGSZ)))
INFER_MAX_MS   = int(os.getenv("ASSISTIVENAV_INFER_MAX_MS", "90"))  # orçamento por inferência

# Warmup pequeno
YOLO_WARMUP_IMGSZ = int(os.getenv("ASSISTIVENAV_YOLO_WARMUP", "224"))

# ---------- TTS ----------
TTS_RATE = int(os.getenv("ASSISTIVENAV_TTS_RATE", "170"))
TTS_VOL  = float(os.getenv("ASSISTIVENAV_TTS_VOL",  "1.0"))
LIST_VOICES     = os.getenv("ASSISTIVENAV_LIST_VOICES", "0") == "1"
CUSTOM_VOICE_ID = os.getenv("ASSISTIVENAV_CUSTOM_VOICE_ID")

# ---------- Talk/HUD ----------
HUD_HOLD_FRAMES   = int(os.getenv("ASSISTIVENAV_HUD_HOLD_FRAMES", "12"))
HUD_FADE_TO_NONE  = os.getenv("ASSISTIVENAV_HUD_FADE", "1") == "1"
HUD_TALK_MIN_HOLD = int(os.getenv("ASSISTIVENAV_HUD_TALK_HOLD", "10"))
TALK_EVT_MIN_HOLD = int(os.getenv("ASSISTIVENAV_EVT_TALK_HOLD", "1"))
TALK_COOLDOWN_S   = float(os.getenv("ASSISTIVENAV_TALK_COOLDOWN", "2.0"))

# ---------- Tracking ----------
CENTER_MATCH_PX       = int(os.getenv("ASSISTIVENAV_CENTER_MATCH_PX", "60"))
MEM_TRACK_TTL         = float(os.getenv("ASSISTIVENAV_MEM_TTL", "1.5"))
MAX_TRACKS_PER_LABEL  = int(os.getenv("ASSISTIVENAV_MAX_TRACKS", "32"))
