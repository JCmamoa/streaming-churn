"""
Modelo 2: Árvore de decisão de churn (gatilhos)
===============================================
Regras interpreteis (max_depth=5) que revelam COMBINAÇÕES de gatilhos
tecnicos + mercadológicos + comportamentais que levam ao cancelamento.
Gera: reports/tree_rules.txt, reports/tree_importance.json, fig árvore.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from common import (load_fact, build_matrix, time_split,
                    classification_metrics, REPORTS)


def main() -> None:
    fact = load_fact()
    X, y = build_matrix(fact)
    X_tr, y_tr, X_te, y_te, _ = time_split(fact, X, y)

    tree = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=800, class_weight=None, random_state=42)
    tree.fit(X_tr, y_tr)
    proba = tree.predict_proba(X_te)[:, 1]
    metrics = {k: round(float(v), 4)
               for k, v in classification_metrics(y_te, proba).items()}

    rules = export_text(tree, feature_names=list(X.columns), decimals=3)
    (REPORTS / "tree_rules.txt").write_text(rules)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(tree, feature_names=list(X.columns), class_names=["fica", "sai"],
              filled=True, max_depth=3, fontsize=7, impurity=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_tree.png", dpi=130)

    imp = pd.DataFrame({"feature": X.columns,
                        "importancia": tree.feature_importances_.round(4)})
    imp = imp.sort_values("importancia", ascending=False)
    imp.to_csv(REPORTS / "tree_importance.csv", index=False)

    payload = {
        "modelo": "Árvore de decisão (churn, depth=5)",
        "metricas_holdout": metrics,
        "importancias": imp.head(10).to_dict(orient="records"),
    }
    (REPORTS / "tree_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"AUC = {metrics['auc']:.3f} | F1 = {metrics['f1']:.3f}")
    print("\nRegras principais salvas em reports/tree_rules.txt")
    print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
