from __future__ import annotations
import os
from core import config
from ultralytics import YOLO
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model():
    if config.PREFER_ONNX:
        if not os.path.isfile(config.YOLO_ONNX_PATH):
            raise FileNotFoundError(f"❌ ONNX não encontrado em {config.YOLO_ONNX_PATH}. Exporte com imgsz={config.YOLO_IMGSZ} e dynamic=False.")
        print(f"✅ ONNX selecionado: {config.YOLO_ONNX_PATH}")
        m = YOLO(config.YOLO_ONNX_PATH)
        print("🚀 Backend: ONNX Runtime (device definido no predict)")
        return m, "onnx"

    # PyTorch fallback (usado só se você desligar PREFER_ONNX)
    if not os.path.isfile(config.YOLO_PT_PATH):
        raise FileNotFoundError(f"❌ Modelo .pt não encontrado em {config.YOLO_PT_PATH}")
    print("✅ Usando modelo .pt (PyTorch).")
    m = YOLO(config.YOLO_PT_PATH)
    try:
        m.to(DEVICE)
    except Exception:
        pass
    try:
        m.model.eval()
    except Exception:
        pass
    print(f"🚀 Modelo pronto (PyTorch) | Device: {DEVICE}")
    return m, "pt"