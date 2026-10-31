# Operação Preditiva — Churn & Win-back | Plataforma de Streaming Musical

Projeto completo de análise preditiva para uma plataforma de streaming musical
com **14% de churn mensal**: identificar os gatilhos tecnológicos, mercadológicos
e comportamentais do cancelamento e calcular a probabilidade de o cliente
**retornar** (win-back).

## Arquitetura (pipeline ETL + modelos)

```
┌────────────────────── EXTRAÇÃO (Rust) ──────────────────────┐
│ log-ingestor: 50M logs/dia do App/Web, agregação user×mês   │
│ src/extraction_rust/ (protótipo de referência + contrato)   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌────────────────── TRANSFORMAÇÃO (Python) ───────────────────┐
│ src/etl/transform.py                                        │
│  • deduplicação de retries de ingestão                      │
│  • imputação de idade média onde faltava                    │
│  • Índice de Frustração Musical (IFM, 0-100)                │
│  • variáveis de preço vs concorrente (scraping semanal)     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────── CARGA (Data Warehouse / Power BI) ───────────┐
│ src/etl/load.py → esquema estrela:                          │
│   DIM_USERS · DIM_MARKET · FACT_MONTHLY · FACT_WINBACK      │
│   (simulação BigQuery; no ambiente real: DirectQuery)       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌────────────────── ANÁLISE PREDITIVA ────────────────────────┐
│ src/models/                                                 │
│  1. logistic_regression.py  → P(cancelar) + odds ratio      │
│  2. decision_tree.py        → regras de gatilhos            │
│  3. combinatorial.py        → combinações de gatilhos/lift  │
│  4. winback.py              → P(retornar) por segmento      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
         reports/REPORT.md (relatório executivo + gráficos)
```

## Fontes de dados (geradas por `src/etl/generate_data.py`)

| Fonte | Tabela | Conteúdo |
|---|---|---|
| Logs App/Web | `panel_monthly` | plays, pulos, tempo de sessão, crashes |
| Transacional | `panel_monthly` | preço do plano, falhas de pagamento |
| Mercado (scraping) | `market_prices` | preços de 3 concorrentes, semanal→mensal |
| CRM/Atendimento | `users` | NPS, chamados, demografia (faixa etária) |
| Campanha/retorno | `campaign`, `winback` | ofertas enviadas, retornos |

50.000 usuários × até 12 meses (~257 mil registros usuário-mês). Semente fixa (42)
→ resultados reprodutíveis. A mecânica causal embutida segue a matriz de decisão
do consumidor: estabilidade técnica, algoritmo de recomendação, sensibilidade a
preço e perfil demográfico afetam realmente a taxa de churn (~13-14% no alvo).

## Como rodar

```bash
# numpy/pandas/etc via uv (https://docs.astral.sh/uv/)
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt

# pipeline completo (E → T → L → modelos → relatório)
.venv/bin/python src/etl/generate_data.py
.venv/bin/python src/etl/transform.py
.venv/bin/python src/etl/load.py
cd src/models
../../.venv/bin/python logistic_regression.py
../../.venv/bin/python decision_tree.py
../../.venv/bin/python combinatorial.py
../../.venv/bin/python winback.py
cd ../reports && ../../.venv/bin/python build_report.py
```

## Resultados principais (holdout temporal meses 11-12)

| Modelo | Métrica | Resultado |
|---|---|---|
| Regressão logística (churn) | AUC | **0.784** |
| Árvore de decisão (gatilhos) | AUC | 0.765 |
| Análise combinatória | pior combinação (3 gatilhos) | churn **48.5%**/mês (lift 3.6x) |
| Win-back | AUC | 0.787 |

- Principal multiplicador de churn: **gap de preço vs concorrente × sensibilidade**
  (odds 2.46) — veja `reports/logistic_metrics.json`.
- Combinações mais letais: `instabilidade + preco_desalinhado + falha_pagamento`
  — veja `reports/combinatoria_gatilhos.csv`.
- Win-back: oferta multiplica por 3-6x a chance de retorno; público 18-24
  é o mais responsivo — `reports/winback_segmentos.csv`.

## Integridade e reprodutibilidade

- `reports/` : métricas JSON, figuras PNG e relatório executivo.
- `data/` : parquet das fontes e do esquema estrela (amostras CSV p/ Power BI).
- Os modelos usam **split temporal** (treino 1-10, validação 11-12) — sem
  vazamento de futuro; o modelo win-back usa log de campanha independente
  do desfecho (sem vazamento de label).
- Dados sintéticos com semente fixa; substitua `src/etl/generate_data.py`
  pela ingestão real (mesmo schema parquet) para operar com dados de produção.
