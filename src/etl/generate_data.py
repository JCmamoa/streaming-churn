"""
Gerador de dados sintéticos - Plataforma de streaming musical
=============================================================
Simula as 4 fontes descritas no projeto:

1. Sistemas internos  -> logs mensais de sessão (tempo, plays, pulos, crashes) e
                          histórico transacional de pagamentos.
2. Dados de mercado   -> precificação de 3 concorrentes (scraping semanal agregado p/ mês).
3. CRM / Atendimento  -> NPS, chamados e dados demográficos do cadastro.
4. Status de vida     -> evento de cancelamento (churn) mensal e retorno (win-back).

A mecânica de geração embute a "matriz de decisão do consumidor":
estabilidade técnica, qualidade do algoritmo, sensibilidade a preço e
perfil demográfico interferem diretamente na probabilidade de churn.
Semente fixa -> resultados reprodutíveis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
N_USERS = 50_000
N_MONTHS = 12
START = pd.Period("2025-01", freq="M")

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_GROUP_W = [0.30, 0.31, 0.19, 0.12, 0.08]
PLANS = ["Premium", "Família", "Estudante"]
PLAN_PRICE = {"Premium": 19.90, "Família": 32.90, "Estudante": 10.90}
COMPETITORS = ["TuneWave", "SoundMax", "RitmoBr"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# 1. CADASTRO (CRM): demografia, plano e sensibilidade a preço
# --------------------------------------------------------------------------
def generate_users() -> pd.DataFrame:
    age_group = RNG.choice(AGE_GROUPS, size=N_USERS, p=AGE_GROUP_W)
    age_ranges = {"18-24": (18, 25), "25-34": (25, 35), "35-44": (35, 45),
                  "45-54": (45, 55), "55+": (55, 75)}
    age_val = np.full(N_USERS, -1.0)
    for g, (lo, hi) in age_ranges.items():
        mask = age_group == g
        age_val[mask] = RNG.integers(lo, hi, mask.sum())

    plan = RNG.choice(PLANS, size=N_USERS, p=[0.62, 0.23, 0.15])
    # ~4% dos cadastros sem idade informada -> será imputado no ETL (idade média)
    age_val[RNG.random(N_USERS) < 0.04] = -1

    # Sensibilidade a preço latente por faixa etária (jovens mais sensíveis)
    sens_map = {"18-24": 1.25, "25-34": 0.90, "35-44": 0.95, "45-54": 1.05, "55+": 1.15}
    sens = np.array([sens_map[g] for g in age_group]) * RNG.normal(1.0, 0.18, N_USERS)

    signup_month = RNG.integers(1, N_MONTHS + 1, size=N_USERS)

    return pd.DataFrame(
        {
            "user_id": np.arange(1, N_USERS + 1),
            "idade": age_val,
            "faixa_etaria": age_group,
            "plano": plan,
            "sensibilidade_preco": sens.round(3),
            "mes_cadastro": signup_month,
        }
    )


# --------------------------------------------------------------------------
# 2. MERCADO: preços semanais dos concorrentes -> média mensal
# --------------------------------------------------------------------------
def generate_market_prices() -> pd.DataFrame:
    rows = []
    for m in range(1, N_MONTHS + 1):
        base = 18.90 + 0.9 * np.sin(m / 2.0)          # ciclos promocionais do mercado
        for c in COMPETITORS:
            for w in range(4):                        # scraping semanal
                price = base + RNG.normal(0, 0.55) + (0.6 if c == "SoundMax" else 0)
                rows.append({"mes": m, "concorrente": c, "semana": w + 1,
                             "preco_premium": round(price, 2)})
    df = pd.DataFrame(rows)
    monthly = (
        df.groupby("mes")["preco_premium"].agg(["min", "mean"])
        .rename(columns={"min": "preco_concorrente_min", "mean": "preco_concorrente_medio"})
        .round(2)
        .reset_index()
    )
    return monthly


# --------------------------------------------------------------------------
# 3. LOGS DE SESSÃO + PAGAMENTOS + CRM + CHURN (painel mensal)
# --------------------------------------------------------------------------
def generate_panel(users: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uid = users["user_id"].to_numpy()
    age_group = users["faixa_etaria"].to_numpy()
    sens = users["sensibilidade_preco"].to_numpy()
    plan = users["plano"].to_numpy()
    signup = users["mes_cadastro"].to_numpy()

    # Qualidade latente por usuário (engajamento estrutural) e trajetória
    engagement = RNG.normal(0.0, 1.0, N_USERS)            # inglês: gosto pelo catálogo/usability
    decay = RNG.normal(0.0, 0.35, N_USERS)                # tendência de desgaste

    records = []
    winbacks = []
    campaigns = []
    churn_events = np.zeros((N_MONTHS + 2, N_USERS), dtype=bool)  # churn_events[m][user]: usuário cancelou no mês m
    state_active = signup <= 0  # começa inativo; ativa no mês de cadastro

    for m in range(1, N_MONTHS + 1):
        newly_active = (signup == m)
        state_active = state_active | newly_active
        # Ativos = quem já entrou e não cancelou em nenhum mês até agora.
        # Retornos de win-back reativam o usuário (churn_events[ret_m] recebe False).
        active_now = state_active & ~churn_events[: m + 1].any(axis=0)

        n = active_now.sum()
        if n == 0:
            continue
        tenure = np.clip(m - signup[active_now], 0, 12)

        # --- Logs do app ---
        plays = RNG.poisson(np.clip(90 + 25 * engagement[active_now] + 2 * tenure, 5, 400))
        skip_rate = np.clip(0.10 + 0.11 * decay[active_now] - 0.02 * engagement[active_now]
                            + np.clip(0.05 * (tenure - 6) / 6, -0.05, 0.05), 0.01, 0.75)
        skips = RNG.binomial(plays, skip_rate)
        session_min = np.clip(RNG.gamma(2.0, 7.5, n) + 6 * engagement[active_now]
                              - 0.5 * skip_rate * 30, 3, 480)
        crash_prob = np.clip(0.05 + 0.04 * (age_group[active_now] == "55+") + RNG.random(n) * 0.05, 0.0, 0.9)
        crashes = RNG.binomial(6, crash_prob)             # crashes no mês

        # --- Mercado / preço ---
        market_row = market.loc[market["mes"] == m].iloc[0]
        plan_price = np.array([PLAN_PRICE[p] for p in plan[active_now]], dtype=float)
        price_gap_pct = (plan_price / market_row["preco_concorrente_min"] - 1.0) * 100

        # --- CRM ---
        latent_nps = engagement[active_now] + 0.4 * (sens[active_now] - 1) * (price_gap_pct > 20) + RNG.normal(0, 1, n)
        nps = np.where(latent_nps > 0.8, 1, np.where(latent_nps < -0.6, -1, 0))
        tickets = RNG.poisson(np.clip(0.4 + 2.2 * np.clip(decay[active_now], 0, None)
                                      + (session_min < 8) * 0.5, 0, 6))
        failed_pay = RNG.random(n) < np.clip(0.02 + 0.10 * (sens[active_now] > 1.2)
                                             + 0.05 * (price_gap_pct > 25), 0, 0.6)

        # --- Score de churn (matriz de decisão) ---
        z = (
            -3.05
            + 0.014 * (skips / np.maximum(plays, 1) * 100)      # pulos / frustração
            + 0.45 * np.log1p(crashes)                          # instabilidade
            + 0.022 * price_gap_pct * sens[active_now]          # preço vs mercado
            + 0.35 * (nps == -1) - 0.20 * (nps == 1)            # NPS detrator/promotor
            + 0.18 * tickets                                    # chamados
            + 0.40 * failed_pay                                 # falha de pagamento
            - 0.08 * tenure                                     # fidelização por tempo
            + 0.35 * (age_group[active_now] == "18-24")         # demografia
            - 0.10 * (age_group[active_now] == "25-34")
            + 0.20 * (age_group[active_now] == "55+")
        )
        churned = RNG.random(n) < _sigmoid(z)
        churn_events[m][uid[active_now][churned]] = True

        records.append(pd.DataFrame(
            {"user_id": uid[active_now], "mes": m, "plays": plays,
             "skips": skips, "session_minutes": session_min,
             "crashes": crashes, "plano": plan[active_now],
             "preco_plano": plan_price,
             "price_gap_pct": price_gap_pct.round(2),
             "nps": nps, "tickets": tickets,
             "failed_payment": failed_pay,
             "churned_event": churned}
        ))

        # --- Win-back: usuários que cancelaram podem retornar ---
        churned_ids = uid[active_now][churned]
        if len(churned_ids) == 0 or m >= N_MONTHS:
            continue
        n_ch = len(churned_ids)

        # Campanha de win-back: decisão de envio de oferta para TODOS que
        # cancelaram (log de campanha) - não depende do desfecho (retorno).
        offer = RNG.random(n_ch) < 0.55
        disc = RNG.choice([10, 20, 30], n_ch) * offer.astype(int)
        campaigns.append(pd.DataFrame({
            "user_id": churned_ids, "mes_churn": m,
            "oferta_enviada": offer, "desconto_pct": disc.astype(float)}))

        # Desfecho: retorno dependendo da oferta e do perfil de churn
        z_wb = (
            -2.55
            + 1.05 * offer                                  # oferta ativa
            + 0.9 * (age_group[active_now][churned] == "18-24")   # jovem volta mais
            - 0.4 * (age_group[active_now][churned] == "55+")
            - 0.35 * np.log1p(skips[churned])               # quem se frustrou, volta menos
            + 0.02 * price_gap_pct[churned] * (offer & (RNG.random(n_ch) < 0.6))
        )
        ret_event = RNG.random(n_ch) < _sigmoid(z_wb)
        ret_m = m + RNG.integers(1, N_MONTHS - m + 1, n_ch)
        sel = ret_event
        winbacks.append(pd.DataFrame({
            "user_id": churned_ids[sel], "mes_churn": m, "mes_retorno": ret_m[sel]}))
        for u, r in zip(churned_ids[sel], ret_m[sel]):
            churn_events[r][u] = False

    panel = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    wb = pd.concat(winbacks, ignore_index=True) if winbacks else pd.DataFrame(
        columns=["user_id", "mes_churn", "mes_retorno"])
    camp = pd.concat(campaigns, ignore_index=True) if campaigns else pd.DataFrame(
        columns=["user_id", "mes_churn", "oferta_enviada", "desconto_pct"])
    return panel, wb, camp


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    users = generate_users()
    market = generate_market_prices()
    panel, winbacks, campaigns = generate_panel(users, market)

    users.to_parquet(DATA_DIR / "users.parquet", index=False)
    market.to_parquet(DATA_DIR / "market_prices.parquet", index=False)
    panel.to_parquet(DATA_DIR / "panel_monthly.parquet", index=False)
    winbacks.to_parquet(DATA_DIR / "winback.parquet", index=False)
    campaigns.to_parquet(DATA_DIR / "campaign.parquet", index=False)

    # Simula retries do ingestor Rust: 0,1% dos logs chegam duplicados
    dup = panel.sample(frac=0.001, random_state=7)
    pd.concat([panel, dup], ignore_index=True).to_parquet(
        DATA_DIR / "panel_monthly.parquet", index=False)

    churn_rate = panel["churned_event"].mean()
    print(f"Usuários: {len(users):,} | Registros no painel: {len(panel):,}")
    print(f"Taxa de churn mensal média (ativa): {churn_rate:.2%}")
    print(f"Eventos de win-back: {len(winbacks):,}")
    print("Arquivos gravados em:", DATA_DIR)


if __name__ == "__main__":
    main()
