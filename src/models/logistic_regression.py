"""
Modelo 1: Regressão Logística-Churn (probabilidade de cancelamento)
===================================================================
Estima P(cancelar no próximo mês) e devolve interpretação gerencial:
coeficientes em odds ratio ("quanto a chance de sair multiplica").
Gera saídas em reports/ (métricas JSON + curva ROC + coeficientes).
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import calibration_curve

from common import (load_fact, build_matrix, time_split,
                    classification_metrics, REPORTS)


def main() -> None:
    fact = load_fact()
    X, y = build_matrix(fact)
    X_tr, y_tr, X_te, y_te, fact_te = time_split(fact, X, y)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    metrics = classification_metrics(y_te, proba)
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    # Coeficientes interpretáveis (odds ratio)
    scaler, lr = model[0], model[1]
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coef": lr.coef_[0].round(4),
        "odds_ratio": np.exp(lr.coef_[0]).round(4),
    }).sort_values("coef", ascending=False)

    # Calibração
    pt, pp = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")

    # Curva ROC
    REPORTS.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_te, proba)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR (1 - especificidade)")
    ax.set_ylabel("TPR (recall)")
    ax.set_title("Regressão Logística - Churn (holdout meses 11-12)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_roc_logit.png", dpi=130)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(pp, pt, "o-", label="Modelo")
    ax2.plot([0, 1], [0, 1], "k--", lw=1)
    ax2.set_xlabel("Probabilidade prevista")
    ax2.set_ylabel("Frequência observada")
    ax2.set_title("Calibração (holdout)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(REPORTS / "fig_calibracao.png", dpi=130)

    # Ranking de risco para sankey de negócio: top 5% risco no holdout
    risk = fact_te[["user_id", "mes"]].copy()
    risk["prob_churn"] = proba.round(4)
    risk.sort_values("prob_churn", ascending=False).head(5000).to_csv(
        REPORTS / "ranking_risco_churn.csv", index=False)

    payload = {
        "modelo": "Regressão Logística (churn mensal)",
        "n_treino": int(len(y_tr)), "n_holdout": int(len(y_te)),
        "taxa_churn_treino": round(float(y_tr.mean()), 4),
        "taxa_churn_holdout": round(float(y_te.mean()), 4),
        "metricas_holdout": metrics,
        "coeficientes": coefs.to_dict(orient="records"),
    }
    (REPORTS / "logistic_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"AUC     = {metrics['auc']:.3f}")
    print(f"Recall  = {metrics['recall']:.3f}")
    print(f"LogLoss = {metrics['log_loss']:.3f}")
    print()
    print(coefs.to_string(index=False))


if __name__ == "__main__":
    main()
