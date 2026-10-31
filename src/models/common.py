"""
Base comum dos modelos preditivos
=================================
- Carrega a tabela analítica do DW (data/dw).
- Define o conjunto de features alinhado à matriz de decisão:
    * Tecnológicos  : IFM, crashes, tempo de sessão
    * Mercadológicos: gap de preço vs concorrente (interação com sensibilidade)
    * Comportamentais/CRM: NPS, chamados, falha de pagamento, tenure
    * Demográficos  : faixa etária (dummies)
- Split temporal: meses 1-10 = treino, 11-12 = validação (holdout futuro).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

NUM_FEATURES = [
    "ifm", "crashes", "session_minutes",
    "price_gap_pct", "tenure_meses", "tickets",
]
BIN_FEATURES = ["failed_payment"]
CAT_FEATURES = ["faixa_etaria"]
TARGET = "churned_event"

FAIXA_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]


def load_fact() -> pd.DataFrame:
    fact = pd.read_parquet(DATA_DIR / "dw" / "FACT_MONTHLY.parquet")
    # Interação preço x sensibilidade (matriz de decisão do consumidor)
    fact["price_gap_x_sens"] = fact["price_gap_pct"] * fact["sensibilidade_preco"]
    fact["nps_det"] = (fact["nps"] == -1).astype(int)
    fact["nps_pro"] = (fact["nps"] == 1).astype(int)
    return fact


def build_matrix(fact: pd.DataFrame):
    X = fact[NUM_FEATURES + BIN_FEATURES].copy()
    X["price_gap_x_sens"] = fact["price_gap_x_sens"]
    X = pd.concat([X, pd.get_dummies(fact["faixa_etaria"], prefix="faixa", dtype=float)],
                  axis=1)
    y = fact[TARGET].astype(int).to_numpy()
    return X, y


def time_split(fact: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
               train_until: int = 10):
    """Split temporal: nunca se treina com o futuro."""
    mask_train = fact["mes"] <= train_until
    return (X[mask_train], y[mask_train.to_numpy()],
            X[~mask_train], y[~mask_train.to_numpy()],
            fact.loc[~mask_train])


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5):
    from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                                 precision_score, recall_score, f1_score)
    pred = (proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)),
        "brier": brier_score_loss(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
