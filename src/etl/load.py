"""
Carga (etapa final do ETL) - consolidação para o Data Warehouse
==============================================================
Simula a conexão com o BigQuery: constrói o esquema estrela consumido
pelo Power BI via DirectQuery.

  dim_users      -> demografia, plano, sensibilidade a preço (CRM)
  dim_market     -> preços mensais dos concorrentes (scraping)
  fact_monthly   -> fato mensal de comportamento + churn + IFM
  fact_winback   -> eventos de retorno pós-cancelamento

Também exporta CSVs de amostra para conferência manual / Power BI Desktop.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SRC = DATA_DIR / "transformed"
OUT = DATA_DIR / "dw"
SCHEMA_VERSION = "1.0"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fact = pd.read_parquet(SRC / "dw_session_events.parquet")
    users = pd.read_parquet(SRC / "dw_users.parquet")
    market = pd.read_parquet(DATA_DIR / "market_prices.parquet")
    winback = pd.read_parquet(SRC / "dw_winback.parquet")
    campaign = pd.read_parquet(DATA_DIR / "campaign.parquet")

    dim_users = users.rename(columns={"faixa_etaria": "faixa_etaria_dim"})
    dim_market = market.copy()

    fact_monthly = fact.copy()
    fact_winback = winback.copy()

    tables = {
        "DIM_USERS": dim_users,
        "DIM_MARKET": dim_market,
        "FACT_MONTHLY": fact_monthly,
        "FACT_WINBACK": fact_winback,
        "FACT_CAMPAIGN": campaign,
    }

    print("=" * 60)
    print(f"Carga para o Data Warehouse (esquema v{SCHEMA_VERSION})")
    print("=" * 60)
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        df.head(1000).to_csv(OUT / f"{name}.csv", index=False)
        print(f"  {name:<14} {len(df):>9,} linhas | {df.shape[1]:>2} colunas")

    # Sanity checks de integridade (equivalente a testes de carga no BigQuery)
    assert fact_monthly["user_id"].notna().all()
    assert fact_monthly["ifm"].between(0, 100).all()
    assert (fact_monthly["price_ratio"] >= 0).all()
    print("\nChecks de integridade: OK")


if __name__ == "__main__":
    main()
