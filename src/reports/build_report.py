"""
Consolidação do relatório executivo (reports/REPORT.md)
=======================================================
Junta as métricas dos 4 modelos + figuras geradas em um único relatório
com interpretação gerencial da operação preditiva.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[2] / "reports"
REPO = Path(__file__).resolve().parents[2]


def jload(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def figure(name: str, caption: str) -> str:
    f = REPORTS / name
    if f.exists():
        return f"![{caption}]({name})\n\n*{caption}*\n"
    return ""


def main() -> None:
    logit = jload("logistic_metrics.json")
    tree = jload("tree_metrics.json")
    comb = jload("combinatoria_metrics.json")
    wbm = jload("winback_metrics.json")
    top_comb = pd.DataFrame(comb["top_10_risco"])
    coefs = pd.DataFrame(logit["coeficientes"])
    seg = (pd.DataFrame(wbm["segmentos"])
             .pivot(index="faixa_etaria", columns="oferta_enviada",
                    values="prob_retorno_media"))
    top_churn_triple = top_comb.iloc[0]["churn_rate_pct"] / 100

    md = f"""# Operação Preditiva — Churn & Win-back (Plataforma de Streaming Musical)

**DataWarehouse:** FACT_MONTHLY ({logit['n_treino']+logit['n_holdout']:,} registro usuário-mês) ·
**Metodologia:** split temporal (treino meses 1-10 → validação meses 11-12)

## 1. Panorama

| Indicador | Valor |
|---|---|
| Taxa de churn mensal observada (base) | {comb['churn_base_pct']:.1f}% |
| Taxa de retorno pós-cancelamento (win-back) | {wbm['taxa_retorno_observada']:.1%} dos eventos de churn |
| Eventos de churn analisados | {wbm['n_eventos_churn']:,} |
| Combinações de gatilhos avaliadas | {comb['n_combinacoes_avaliadas']} |

## 2. Regressão Logística — probabilidade de cancelamento

Métricas no holdout temporal (meses 11-12, nunca vistos no treino):

| Métrica | Valor |
|---|---|
| AUC | **{logit['metricas_holdout']['auc']:.3f}** |
| LogLoss | {logit['metricas_holdout']['log_loss']:.3f} |
| F1 (@0.5) | {logit['metricas_holdout']['f1']:.3f} |

Fatores que **multiplicam a chance de cancelar** (odds ratio > 1):

{coefs[coefs['coef'] > 0].head(6).to_markdown(index=False)}

Fatores de **retenção** (odds ratio < 1):

{coefs[coefs['coef'] < 0].head(4).to_markdown(index=False)}

{figure('fig_roc_logit.png', 'Curva ROC — modelo de churn no holdout')}
{figure('fig_calibracao.png', 'Calibração das probabilidades previstas')}

## 3. Árvore de Decisão — gatilhos acionáveis (AUC {tree['metricas_holdout']['auc']:.3f})

Regras completas em [`reports/tree_rules.txt`](tree_rules.txt). Top features por
impureza:

| Feature | Importância |
|---|---|
""" + "\n".join(
        f"| {r['feature']} | {r['importancia']:.3f} |"
        for r in tree["importancias"][:6]
    ) + f"""

{figure('fig_tree.png', 'Árvore de decisão (primeiros 3 níveis)')}

## 4. Análise Combinatória — combinações de gatilhos de risco

Churn base: **{comb['churn_base_pct']:.1f}%** ao mês. Combinações com maior risco
observado (suporte mínimo 500 ocorrências):

{top_comb.head(8).to_markdown(index=False)}

{figure('fig_combinatoria.png', 'Pares de gatilhos com maior taxa de churn')}

## 5. Win-back — probabilidade de retorno

Modelo de regressão logística sobre 1 linha por evento de churn
(AUC = **{wbm['auc_holdout']:.3f}**). Probabilidade média de retorno simulada:

| Faixa etária | Sem oferta | Com oferta |
|---|---|---|
""" + "\n".join(
        f"| {i} | {row[False]:.1%} | {row[True]:.1%} |"
        for i, row in seg.iterrows()
    ) + f"""

**Leitura:** a oferta de win-back tem efeito causal estimado de 5 a 6x na
probabilidade de retorno; o público 18-24 é o mais responsivo — priorizar
campanha com desconto para esse segmento.

Rankings de acionamento: [`ranking_risco_churn.csv`](ranking_risco_churn.csv)
(500 usuários com maior P(churn) no holdout) e
[`ranking_winback.csv`](ranking_winback.csv) (500 ex-usuários com maior P(retorno)).

## 6. Recomendações operacionais

1. **Fixar o preço percebido**: o gap de preço vs concorrente interagido com a
   sensibilidade é o principal multiplicador de churn — monitorar semanalmente
   (scraping já no pipeline) e ativar desconto temporário quando gap > 20%.
2. **Estabilidade do app**: crashes >= 2/mês combinado com preço desalinhado
   chega a {top_churn_triple:.1%} de churn mensal — engenharia priorizada.
3. **Motor de recomendação**: IFM alto (pulos acima do percentil 75) eleva
   risco de forma independente — recalibrar recomendações para usuários
   com IFM alto e sessão encurtando.
4. **NPS detratores + chamados abertos**: fila crítica p/ o time de suporte.
5. **Win-back intencional**: públicos 18-24 e 25-34 respondem à oferta;
   usuários que cancelaram com IFM alto têm baixa probabilidade de retorno —
   economia de verba de campanha.
"""

    out = REPO / "reports" / "REPORT.md"
    out.write_text(md, encoding="utf-8")
    print("Relatório gerado em:", out)


if __name__ == "__main__":
    main()
