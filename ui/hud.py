# ui/hud.py
from __future__ import annotations

class HudStabilizer:
    def __init__(self, hold_frames=12, fade_to_none=True, empty_text="Nenhum objeto"):
        self.empty_text = empty_text
        self.last_text = empty_text
        self.hold = 0
        self.hold_frames = int(hold_frames)
        self.fade_to_none = bool(fade_to_none)

    def update(self, new_text: str | None):
        """
        Se new_text == None: mantém o texto por 'hold' frames e depois
        volta para empty_text (se fade_to_none=True).
        Se new_text != None: troca o texto e recomeça o hold.
        """
        if new_text is None:
            if self.hold > 0:
                self.hold -= 1
                return self.last_text
            else:
                return self.empty_text if self.fade_to_none else self.last_text
        else:
            if new_text != self.last_text:
                self.last_text = new_text
            self.hold = self.hold_frames
            return self.last_text

    def clear(self):
        """
        Limpa imediatamente o HUD, cancelando o hold e voltando para o texto vazio.
        Compatível com app.py que chama hud.clear().
        """
        self.last_text = self.empty_text
        self.hold = 0

    # (Opcional) se quiser uma API mais explícita:
    def set_text(self, text: str, hold_frames: int | None = None):
        """
        Define o texto atual e reinicia o hold. Útil para forçar um HUD específico.
        """
        self.last_text = text or self.empty_text
        if hold_frames is not None:
            self.hold_frames = int(hold_frames)
        self.hold = self.hold_frames
