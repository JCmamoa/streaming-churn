"""
Transformação e Limpeza (etapa ETL - Python)
============================================
Recebe os dados das fontes sintéticas como se fossem extraídos pelos
microserviços Rust e produz a tabela analítica que alimentaria o Data
Warehouse (BigQuery -> Power BI via DirectQuery).

Etapa T do ETL:
- Deduplicação de registros de log (idempotência da ingestão).
- Tratamento de nulos  -> imputação de idade média onde faltava.
- Renomeação de valores inconsistentes (padronização de faixas etárias).
- Variáveis calculadas:
    * skip_rate            : % de pulos sobre reproduções
    * Índice de Frustração Musical (IFM) : score 0-100 combinando pulos,
      saltos em sequência sugerida e deterioração de sessão.
    * tenure_meses         : tempo de casa no mês da observação
    * revisitado: preço do plano vs mercado em janela móvel de 4 semanas
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR = DATA_DIR / "transformed"

PLANO_ALIASES = {"PREMIUM": "Premium", "premium ": "Premium", "FAMILIA": "Família"}


def dedupe_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Remove registros duplicados de (user_id, mes) recebidos duas vezes
    pelo ingestor (retries do microserviço Rust)."""
    before = len(panel)
    panel = panel.drop_duplicates(subset=["user_id", "mes"], keep="last")
    print(f"[dedupe] {before - len(panel)} registros duplicados removidos "
          f"({before:,} -> {len(panel):,})")
    return panel


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Padroniza categorias e imputa idade média por faixa etária."""
    users = users.copy()
    users["plano"] = users["plano"].str.strip().replace(PLANO_ALIASES)
    users["idade"] = pd.to_numeric(users["idade"], errors="coerce")

    mask_invalid = ~users["idade"].between(10, 120)
    print(f"[idade] {mask_invalid.sum()} cadastros sem idade válida -> imputação pela média")

    # Imputação de idade média por faixa etária (caso não tenha faixa, média global)
    global_mean = users.loc[~mask_invalid, "idade"].mean()
    users.loc[mask_invalid, "idade"] = (
        users.loc[mask_invalid, "faixa_etaria"]
        .map(users.groupby("faixa_etaria")["idade"].mean())
        .fillna(global_mean)
    ).round(0)
    return users


def compute_frustration_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Índice de Frustração Musical (IFM), 0-100.

    Pesos:
      - 55% : taxa de pulos sobre reproduções (o sinal mais forte)
      - 25% : pulos 'atacado' (>= 50% das reproduções puladas - zapping)
      - 20% : deterioração de engajamento (queda de sessão média)
    """
    df = panel.sort_values(["user_id", "mes"]).copy()
    df["skip_rate"] = df["skips"] / df["plays"].clip(lower=1)

    df["ifm_base"] = (df["skip_rate"] * 100).clip(0, 100) * 0.55
    df["ifm_zapping"] = (df["skip_rate"] >= 0.5).astype(float) * 25.0

    g = df.groupby("user_id")["session_minutes"]
    df["session_diff"] = g.diff().fillna(0.0)
    max_drop = df.groupby("user_id")["session_diff"].min().clip(-60, 0) * -1  # maior queda
    df = df.merge(max_drop.rename("session_max_drop"), on="user_id", how="left")
    df["ifm_engagement"] = (df["session_max_drop"] / 60 * 20.0).clip(0, 20)

    df["ifm"] = (df["ifm_base"] + df["ifm_zapping"] + df["ifm_engagement"]).round(1).clip(0, 100)
    return df.drop(columns=["ifm_base", "ifm_zapping", "ifm_engagement", "session_diff",
                            "session_max_drop"])


def add_features(panel: pd.DataFrame, users: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()

    # Demografia do cadastro
    df = df.merge(users[["user_id", "idade", "faixa_etaria", "sensibilidade_preco"]],
                  on="user_id", how="left")

    # Tenure (tempo de casa em meses no mês da observação)
    signup = users.set_index("user_id")["mes_cadastro"]
    df["tenure_meses"] = df["user_id"].map(signup)
    df["tenure_meses"] = (df["mes"] - df["tenure_meses"]).clip(0, 12)

    # Contexto de mercado (preço mínimo do concorrente no mês do observação)
    df = df.merge(market[["mes", "preco_concorrente_min", "preco_concorrente_medio"]],
                  on="mes", how="left")
    df["price_ratio"] = df["preco_plano"] / df["preco_concorrente_min"]
    df["price_gap_pct"] = (df["price_ratio"] - 1) * 100

    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    users = pd.read_parquet(DATA_DIR / "users.parquet")
    panel = pd.read_parquet(DATA_DIR / "panel_monthly.parquet")
    market = pd.read_parquet(DATA_DIR / "market_prices.parquet")
    winback = pd.read_parquet(DATA_DIR / "winback.parquet")
    campaign = pd.read_parquet(DATA_DIR / "campaign.parquet")

    print("=" * 60)
    print("ETL - Transformação e Limpeza")
    print("=" * 60)
    panel = dedupe_panel(panel)
    users = clean_users(users)
    panel = compute_frustration_index(panel)
    panel = add_features(panel, users, market)

    cols = ["user_id", "mes", "plays", "skips", "session_minutes", "crashes",
            "plano", "preco_plano", "price_gap_pct", "price_ratio",
            "nps", "tickets", "failed_payment",
            "skip_rate", "ifm", "idade", "faixa_etaria",
            "sensibilidade_preco", "tenure_meses", "churned_event"]
    panel = panel[cols]

    panel.to_parquet(OUT_DIR / "dw_session_events.parquet", index=False)
    winback.to_parquet(OUT_DIR / "dw_winback.parquet", index=False)
    campaign.to_parquet(OUT_DIR / "dw_campaign.parquet", index=False)
    users.to_parquet(OUT_DIR / "dw_users.parquet", index=False)

    print("\n[IFM] Descrição:")
    print(panel["ifm"].describe().round(2).to_string())
    print("\nTabela analítica gravada (simulando BigQuery/Power BI):")
    for f in sorted(OUT_DIR.glob("*.parquet")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
