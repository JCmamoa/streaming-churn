"""
Modelo 3: Análise combinatória de gatilhos
==========================================
Transforma os gatilhos contínuos da matriz de decisão em fatores binários
(presença/ausência) e avalia TODAS as combinações de 1 a K fatores:

  - suporte       : % dos registros usuário-mês que exibem a combinação
  - churn         : taxa de cancelamento observada na combinação
  - lift          : churn da combinação / churn base
  - risco additivo: contribuição incremental ao juntar fatores

Metodologia faz o papel das "análises combinatórias" de gatilhos usadas
quando se quer regras de acionamento (ex.: campanha de retenção) sem
depender de um modelo único.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from common import load_fact, REPORTS

MAX_K = 3
PERCENT_THRESHOLD = 0.60  # padrão precisa ter no mínimo 60% de ganho de risco

TRIGGER_DEFS = {
    "IFM_alto":            lambda f: f["ifm"] >= f["ifm"].quantile(0.75),
    "instabilidade":       lambda f: f["crashes"] >= 2,
    "preco_desalinhado":   lambda f: f["price_gap_pct"] * f["sensibilidade_preco"] >= 20,
    "detrator_NPS":        lambda f: f["nps"] == -1,
    "chamados_atacado":    lambda f: f["tickets"] >= 2,
    "falha_pagamento":     lambda f: f["failed_payment"] == 1,
    "sessao_curta":        lambda f: f["session_minutes"] <= 10,
    "jovem_18_24":         lambda f: f["faixa_etaria"] == "18-24",
}


def build_triggers(fact: pd.DataFrame) -> pd.DataFrame:
    trig = pd.DataFrame(index=fact.index)
    for name, fn in TRIGGER_DEFS.items():
        trig[name] = fn(fact).astype(int).to_numpy()
    trig["churned"] = fact["churned_event"].astype(int).to_numpy()
    return trig


def evaluate_combinations(trig: pd.DataFrame, max_k: int = MAX_K) -> pd.DataFrame:
    names = list(TRIGGER_DEFS.keys())
    base_churn = trig["churned"].mean()
    rows = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(names, k):
            mask = np.ones(len(trig), dtype=bool)
            for c in combo:
                mask &= trig[c].to_numpy() == 1
            n = mask.sum()
            if n < 500:  # suporte mínimo p/ confiabilidade estatística
                continue
            churn_comb = trig.loc[mask, "churned"].mean()
            rows.append({
                "k_fatores": k,
                "combinacao": " + ".join(combo),
                "n_observacoes": int(n),
                "suporte_pct": round(100 * n / len(trig), 2),
                "churn_rate_pct": round(100 * churn_comb, 2),
                "lift_vs_base": round(churn_comb / base_churn, 2),
            })
    return (pd.DataFrame(rows)
            .sort_values("churn_rate_pct", ascending=False))


def main() -> None:
    fact = load_fact()
    trig = build_triggers(fact)
    base = trig["churned"].mean()
    print(f"Churn base do período: {base:.2%}")
    print(f"Prevalência de cada gatilho:")
    for name in TRIGGER_DEFS:
        print(f"  - {name:<18} {trig[name].mean():.1%}")

    results = evaluate_combinations(trig)
    results.to_csv(REPORTS / "combinatoria_gatilhos.csv", index=False)

    # Heatmap: churn por pares de gatilhos mais representativos
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = results[results["k_fatores"] == 2].head(15)
    labels = [c.replace(" + ", "\n+ ") for c in pairs["combinacao"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(pairs)), pairs["churn_rate_pct"], color="#d33")
    ax.axhline(100 * base, ls="--", color="k", lw=1,
               label=f"churn base = {100*base:.1f}%")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Taxa de churn na combinação (%)")
    ax.set_title("Análise combinatória - pares de gatilhos com maior risco")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_combinatoria.png", dpi=130)

    payload = {
        "churn_base_pct": round(100 * base, 2),
        "n_combinacoes_avaliadas": int(len(results)),
        "top_10_risco": results.head(10).to_dict(orient="records"),
    }
    (REPORTS / "combinatoria_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\nTotal de combinações avaliadas (k=1..{MAX_K}): {len(results)}")
    print("\nTOP 10 combinações por taxa de churn:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
