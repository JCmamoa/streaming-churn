# Extrator de Logs — Rust (camada Extract do ETL)

Microserviço de referência para ingestão dos ~50M logs/dia do app/web.

## Papel no pipeline

```
App/Web  ──(NDJSON stream)──▶  log-ingestor (Rust)  ──▶  Kafka/Batch  ──▶  Python (T)  ──▶  BigQuery (L)
```

- **Entrada**: eventos brutos por linha (NDJSON):
  `{"user_id":123,"mes":4,"plays":60,"skips":8,"session_minutes":31.5,"crashes":1}`
- **Saída**: agregado por usuário×mês — exatamente o formato consumido por
  `src/etl/transform.py` na etapa de Transformação.
- **Robustez**: payload malformado é descartado e contabilizado, sem derrubar o serviço
  (nada de panic em fluxo de dados).

## Por que Rust nesta camada

1. **Vazão**: parsing de JSON + agregação em memória a ~GB/s por nó, sem GC.
2. **Segurança de memória**: buffers zero-copy; impossível estourar memória
   silenciosamente sob pico de 50M eventos/dia.
3. **Custo**: menos nós de ingestão comparado a alternativas interpretadas.

## Como testar localmente (protótipo)

```bash
cargo run < exemplo.ndjson          # consome eventos de um arquivo
# ou
cargo build --release              # binário único, sem runtime externo
```

## Contrato de eventos (acordo com o app)

| campo            | tipo  | regra de validação |
|------------------|-------|--------------------|
| `user_id`        | u64   | > 0                |
| `mes`            | u8    | 1..=12             |
| `plays`/`skips`  | u32   | skips <= plays (corrigido na etapa T) |
| `session_minutes`| f32   | clamp 0..480 na etapa T |
| `crashes`        | u8    | agregado na etapa T |

## Status

**Protótipo de referência** — valida o contrato e a lógica de agregação que o
resto do projeto consome. Para produção, acrescentar: cliente Kafka/Redpanda,
backpressure, métricas Prometheus e checkpoints para reprocessamento idempotente
(a deduplicação por `(user_id, mes)` já está implementada na etapa T do Python).
