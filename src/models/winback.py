"""
Modelo 4: Win-back - probabilidade de retorno pós-cancelamento
==============================================================
Para cada usuário que cancelou, estima P(retornar). Usa como features
o estado do usuário NO MOMENTO do cancelamento (última observação antes
do churn) + информаções da campanha de win-back (oferta/desconto).

Saídas:
- Métricas (AUC etc.) em reports/winback_metrics.json
- Tabela segmentada de probabilidade média de retorno por
  faixa etária x oferta (reports/winback_segmentos.csv)
- Ranking dos ex-usuários com maior P(retorno) p/ campanha
  (reports/ranking_winback.csv)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from common import load_fact, DATA_DIR, REPORTS


def churn_snapshot(fact: pd.DataFrame) -> pd.DataFrame:
    """Estado do usuário no mês do cancelamento."""
    ev = fact[fact["churned_event"]].copy()           # uma linha por evento de churn
    return ev


def main() -> None:
    fact = load_fact()
    wb = pd.read_parquet(DATA_DIR / "dw" / "FACT_CAMPAIGN.parquet")
    snap = churn_snapshot(fact)

    # Alvo: o usuário retornou após cancelar
    returns = pd.read_parquet(DATA_DIR / "dw" / "FACT_WINBACK.parquet")
    returned = set(zip(returns["user_id"], returns["mes_churn"]))
    snap["target_return"] = [
        (u, m) in returned for u, m in zip(snap["user_id"], snap["mes"])
    ]

    # Campanha de win-back registrada no CRM para CADA churned (sem vazamento:
    # o envio da oferta não depende do desfecho)
    oferta = wb[["user_id", "mes_churn", "oferta_enviada", "desconto_pct"]]
    snap = snap.merge(oferta, left_on=["user_id", "mes"],
                      right_on=["user_id", "mes_churn"], how="left")
    snap["oferta_enviada"] = snap["oferta_enviada"].fillna(False).astype(float)

    feats = ["ifm", "crashes", "price_gap_pct", "sensibilidade_preco",
             "tickets", "failed_payment", "session_minutes", "tenure_meses"]
    X = snap[feats].copy()
    X["idade"] = snap["idade"] if "idade" in snap else np.nan
    for f in ["18-24", "25-34", "35-44", "45-54", "55+"]:
        X[f"faixa_{f}"] = (snap["faixa_etaria"] == f).astype(float)
    X["oferta"] = snap["oferta_enviada"].to_numpy()  # feature da campanha

    y = snap["target_return"].astype(int).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                              random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    # Coeficientes interpretáveis
    scaler, lr = model[0], model[1]
    coefs = pd.DataFrame({"feature": X.columns,
                          "odds_ratio": np.exp(lr.coef_[0]).round(4)})
    coefs.to_csv(REPORTS / "winback_odds.csv", index=False)

    # Segmentação de negócio: P(retorno) médio por faixa etária e oferta
    # (cenário de simulação: com e sem campanha de win-back)
    seg_rows = []
    for faixa in ["18-24", "25-34", "35-44", "45-54", "55+"]:
        for oferta, z_oferta in [(False, 0.0), (True, 1.0)]:
            xs = X[X[f"faixa_{faixa}"] == 1].copy()
            xs["oferta"] = z_oferta
            p = model.predict_proba(xs)[:, 1].mean()
            seg_rows.append({"faixa_etaria": faixa,
                             "oferta_enviada": oferta,
                             "prob_retorno_media": round(float(p), 4),
                             "n_ex_usuarios": int(len(xs))})
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(REPORTS / "winback_segmentos.csv", index=False)

    # Ranking de campanha intensiva
    ranking = snap[["user_id", "mes", "faixa_etaria", "ifm",
                    "price_gap_pct"]].copy()
    ranking["prob_retorno"] = model.predict_proba(X)[:, 1].round(4)
    ranking.sort_values("prob_retorno", ascending=False).head(5000).to_csv(
        REPORTS / "ranking_winback.csv", index=False)

    payload = {
        "modelo": "Regressão logística win-back",
        "n_eventos_churn": int(len(y)),
        "taxa_retorno_observada": round(float(y.mean()), 4),
        "auc_holdout": round(float(auc), 4),
        "segmentos": seg.to_dict(orient="records"),
    }
    (REPORTS / "winback_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"AUC win-back = {auc:.3f} | Retorno observado = {y.mean():.1%}")
    print(seg.pivot(index="faixa_etaria", columns="oferta_enviada",
                    values="prob_retorno_media").round(3).to_string())


if __name__ == "__main__":
    main()
