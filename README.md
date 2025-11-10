```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```


## Execução
```bash
export OPENAI_API_KEY="sua_chave" # opcional (Linux/macOS)
setx OPENAI_API_KEY "sua_chave" # opcional (Windows)
python -m app.py
```


## Configuração via variáveis de ambiente
- `ASSISTIVENAV_CAMERA_URL` (default: IP Webcam `/video`)
- `ASSISTIVENAV_PROC_EVERY` (frames entre inferências)
- `ASSISTIVENAV_PREFER_ONNX` (1 para tentar ONNX)
- `ASSISTIVENAV_YOLO_PT`, `ASSISTIVENAV_YOLO_ONNX`
- `ASSISTIVENAV_TTS_RATE`, `ASSISTIVENAV_TTS_VOL`, `ASSISTIVENAV_CUSTOM_VOICE_ID`


## Estrutura
- `core/` → utilitários (config, TTS, heurísticas, smoothing, tracking)
- `ui/` → HUD e política de fala
- `vision/` → carregamento YOLO (PT/ONNX)
- `llm/` → assinatura DSPy + predictor
- `app.py` → loop principal
