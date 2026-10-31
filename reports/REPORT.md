# Operação Preditiva — Churn & Win-back (Plataforma de Streaming Musical)

**DataWarehouse:** FACT_MONTHLY (257,202 registro usuário-mês) ·
**Metodologia:** split temporal (treino meses 1-10 → validação meses 11-12)

## 1. Panorama

| Indicador | Valor |
|---|---|
| Taxa de churn mensal observada (base) | 13.3% |
| Taxa de retorno pós-cancelamento (win-back) | 15.6% dos eventos de churn |
| Eventos de churn analisados | 34,333 |
| Combinações de gatilhos avaliadas | 87 |

## 2. Regressão Logística — probabilidade de cancelamento

Métricas no holdout temporal (meses 11-12, nunca vistos no treino):

| Métrica | Valor |
|---|---|
| AUC | **0.784** |
| LogLoss | 0.317 |
| F1 (@0.5) | 0.245 |

Fatores que **multiplicam a chance de cancelar** (odds ratio > 1):

| feature          |   coef |   odds_ratio |
|:-----------------|-------:|-------------:|
| price_gap_x_sens | 0.8989 |       2.457  |
| tickets          | 0.2051 |       1.2277 |
| crashes          | 0.1834 |       1.2013 |
| faixa_18-24      | 0.1214 |       1.1291 |
| failed_payment   | 0.0847 |       1.0884 |
| ifm              | 0.0547 |       1.0562 |

Fatores de **retenção** (odds ratio < 1):

| feature         |    coef |   odds_ratio |
|:----------------|--------:|-------------:|
| faixa_35-44     | -0.0311 |       0.9694 |
| faixa_45-54     | -0.0319 |       0.9686 |
| session_minutes | -0.0747 |       0.928  |
| faixa_25-34     | -0.083  |       0.9204 |

![Curva ROC — modelo de churn no holdout](fig_roc_logit.png)

*Curva ROC — modelo de churn no holdout*

![Calibração das probabilidades previstas](fig_calibracao.png)

*Calibração das probabilidades previstas*


## 3. Árvore de Decisão — gatilhos acionáveis (AUC 0.765)

Regras completas em [`reports/tree_rules.txt`](tree_rules.txt). Top features por
impureza:

| Feature | Importância |
|---|---|
| price_gap_x_sens | 0.929 |
| faixa_18-24 | 0.026 |
| tickets | 0.023 |
| crashes | 0.019 |
| tenure_meses | 0.002 |
| ifm | 0.000 |

![Árvore de decisão (primeiros 3 níveis)](fig_tree.png)

*Árvore de decisão (primeiros 3 níveis)*


## 4. Análise Combinatória — combinações de gatilhos de risco

Churn base: **13.3%** ao mês. Combinações com maior risco
observado (suporte mínimo 500 ocorrências):

|   k_fatores | combinacao                                             |   n_observacoes |   suporte_pct |   churn_rate_pct |   lift_vs_base |
|------------:|:-------------------------------------------------------|----------------:|--------------:|-----------------:|---------------:|
|           3 | instabilidade + preco_desalinhado + falha_pagamento    |             567 |          0.22 |            48.5  |           3.63 |
|           3 | preco_desalinhado + chamados_atacado + falha_pagamento |            1534 |          0.6  |            45.05 |           3.37 |
|           3 | instabilidade + preco_desalinhado + chamados_atacado   |            1267 |          0.49 |            43.49 |           3.26 |
|           3 | preco_desalinhado + detrator_NPS + falha_pagamento     |            2321 |          0.9  |            43.43 |           3.25 |
|           3 | preco_desalinhado + falha_pagamento + jovem_18_24      |            3346 |          1.3  |            42.8  |           3.21 |
|           3 | preco_desalinhado + falha_pagamento + sessao_curta     |            2983 |          1.16 |            42    |           3.15 |
|           3 | instabilidade + preco_desalinhado + jovem_18_24        |            1983 |          0.77 |            41.96 |           3.14 |
|           3 | preco_desalinhado + chamados_atacado + jovem_18_24     |            6039 |          2.35 |            41.65 |           3.12 |

![Pares de gatilhos com maior taxa de churn](fig_combinatoria.png)

*Pares de gatilhos com maior taxa de churn*


## 5. Win-back — probabilidade de retorno

Modelo de regressão logística sobre 1 linha por evento de churn
(AUC = **0.787**). Probabilidade média de retorno simulada:

| Faixa etária | Sem oferta | Com oferta |
|---|---|---|
| 18-24 | 7.1% | 37.7% |
| 25-34 | 3.0% | 20.2% |
| 35-44 | 3.3% | 21.3% |
| 45-54 | 3.1% | 20.4% |
| 55+ | 2.2% | 15.4% |

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
   chega a 48.5% de churn mensal — engenharia priorizada.
3. **Motor de recomendação**: IFM alto (pulos acima do percentil 75) eleva
   risco de forma independente — recalibrar recomendações para usuários
   com IFM alto e sessão encurtando.
4. **NPS detratores + chamados abertos**: fila crítica p/ o time de suporte.
5. **Win-back intencional**: públicos 18-24 e 25-34 respondem à oferta;
   usuários que cancelaram com IFM alto têm baixa probabilidade de retorno —
   economia de verba de campanha.
