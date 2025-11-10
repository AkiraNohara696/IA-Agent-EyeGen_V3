# ui/talk_gate.py
from __future__ import annotations
import time

class TalkGate:
    """
    Controlador de fala com histerese (segurar frames) + cooldown por mensagem.
    - Evita falar textos instáveis (precisa segurar 'min_hold_frames' frames).
    - Evita repetir o mesmo texto dentro do 'cooldown_s'.
    """
    def __init__(self, min_hold_frames: int = 10, cooldown_s: float = 2.0):
        self.min_hold_frames = int(min_hold_frames)
        self.cooldown_s = float(cooldown_s)
        self._cur_text: str | None = None
        self._hold: int = 0
        self._last_spoken: dict[str, float] = {}  # texto -> timestamp

    def clear(self) -> None:
        """Zera estado para quando a cena fica vazia/escura (evita ‘grudar’ fala antiga)."""
        self._cur_text = None
        self._hold = 0

    def update_and_should_speak(self, text: str | None) -> bool:
        t = time.time()
        txt = (text or "").strip()
        if not txt:
            self._cur_text = None
            self._hold = 0
            return False

        if txt != self._cur_text:
            self._cur_text = txt
            self._hold = 1
            return False
        else:
            self._hold += 1

        if self._hold < self.min_hold_frames:
            return False

        last = self._last_spoken.get(txt, 0.0)
        if (t - last) < self.cooldown_s:
            return False

        self._last_spoken[txt] = t
        return True
