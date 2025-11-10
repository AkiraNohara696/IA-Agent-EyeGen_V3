from __future__ import annotations
import pyttsx3
from typing import Optional
from core import config

_TTS_ENGINE: Optional[pyttsx3.Engine] = None


def init_tts() -> Optional[pyttsx3.Engine]:
    """Inicializa TTS em modo não-bloqueante e tenta voz PT-BR automaticamente."""
    global _TTS_ENGINE
    if _TTS_ENGINE is not None:
        return _TTS_ENGINE
    try:
        try:
            engine = pyttsx3.init(driverName='sapi5')
        except Exception:
            engine = pyttsx3.init()
        engine.setProperty('rate', config.TTS_RATE)
        engine.setProperty('volume', config.TTS_VOL)
        try:
            voices = engine.getProperty('voices')
            if config.LIST_VOICES:
                print("=== VOZES DISPONÍVEIS ===")
                for v in voices:
                    print("ID:", v.id, "| Nome:", v.name, "| Langs:", getattr(v, 'languages', []))
                print("=========================")
            if config.CUSTOM_VOICE_ID:
                engine.setProperty('voice', config.CUSTOM_VOICE_ID)
            else:
                chosen = None
                for v in voices:
                    langs = getattr(v, "languages", [])
                    lang_code = ""
                    if langs:
                        lang_code = (langs[0].decode().lower() if isinstance(langs[0], (bytes, bytearray)) else str(langs[0]).lower())
                    if "pt" in lang_code or "port" in (v.name or "").lower() or "brazil" in (v.name or "").lower():
                        chosen = v.id
                        break
                if chosen:
                    engine.setProperty('voice', chosen)
        except Exception as e:
            print("⚠️ TTS: não consegui configurar voz:", e)
        engine.startLoop(False)
        _TTS_ENGINE = engine
        return engine
    except Exception as e:
        print(f"⚠️ Erro ao inicializar TTS: {e}")
        return None


def falar(texto: str):
    """Agenda fala sem bloquear vídeo e sem cortar locução atual."""
    if _TTS_ENGINE:
        try:
            if not _TTS_ENGINE.isBusy():
                _TTS_ENGINE.say(texto)
        except Exception as e:
            print("⚠️ Erro no TTS:", e)


def iterate():
    if _TTS_ENGINE:
        try:
            _TTS_ENGINE.iterate()
        except Exception:
            pass


def end_loop():
    if _TTS_ENGINE:
        try:
            _TTS_ENGINE.endLoop()
        except Exception:
            pass