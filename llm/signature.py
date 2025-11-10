from __future__ import annotations
import os
import dspy
from typing import Optional
from core import config


class AssistiveNavigation(dspy.Signature):
    """
    Você é um agente de IA assistiva para guiar uma pessoa com deficiência visual.
    Regras:
    - Responda em PT-BR.
    - UMA linha: DIREÇÃO; DISTÂNCIA; MOTIVO.
    - Direções: 'siga em frente' | 'desvie para a esquerda' | 'desvie para a direita' | 'pare'.
    - Distâncias: 0,5 m | 1 m | 2 m | 3 m | 5 m.
    - Se incerto, diga 'pare' e peça confirmação.
    """
    contexto = dspy.InputField(desc="Descrição curta do ambiente e detecções")
    alvo = dspy.InputField(desc="Nome do objeto mais relevante (ex.: pessoa, cadeira)")
    centralidade = dspy.InputField(desc="Desvio do centro (-1 esquerda, 0 centro, +1 direita)")
    proximidade = dspy.InputField(desc="Estimativa de distância em metros (float)")
    saida = dspy.OutputField(desc="Ex.: desvie para a direita; 1 m; pessoa à frente")


_predict_nav: Optional[dspy.Predict] = None


def init_llm():
    global _predict_nav
    if not config.USE_LLM or not config.OPENAI_KEY:
        dspy.configure(lm=None)
        _predict_nav = None
        return None
    try:
        from dspy.utils.openai import OpenAI
        dspy.configure(
            lm=OpenAI(
                model=config.LLM_MODEL,
                api_key=config.OPENAI_KEY,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
        )
        _predict_nav = dspy.Predict(AssistiveNavigation)
        return _predict_nav
    except Exception:
        dspy.configure(lm=None)
        _predict_nav = None
        return None


def predict_line(contexto: str, alvo: str, centralidade: float, proximidade: float) -> str | None:
    if _predict_nav is None:
        return None
    try:
        resp = _predict_nav(
            contexto=contexto,
            alvo=alvo,
            centralidade=str(round(float(centralidade), 2)),
            proximidade=str(float(proximidade)),
        )
        line = (resp.saida or "").strip()
        return line or None
    except Exception:
        return None