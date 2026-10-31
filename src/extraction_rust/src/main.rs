//! Microserviço de ingestão de logs (camada Extract do ETL)
//! =========================================================
//! Em produção: consome ~50M de logs/dia do app e da web em tempo real,
//! valida o schema, agrega por usuário/mês e publica no pipeline
//! (Kafka -> BigQuery). Aqui: protótipo de referência com o contrato
//! de eventos e a lógica de agregação que alimenta os modelos preditivos.
//!
//! Por que Rust: ingestão de alto volume com uso de memória previsível
//! (zero-copy de buffers, sem GC), latência baixa e segurança frente a
//! payloads malformados.

use serde::Deserialize;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

/// Evento bruto de log emitido pelo app/web.
/// Contrato compartilhado com a camada Python de transformação:
///   user_id, mes, plays, skips, session_minutes, crashes
#[derive(Debug, Deserialize)]
struct LogEvent {
    user_id: u64,
    /// mês de referência (1..=12) — agregação mensal
    mes: u8,
    plays: u32,
    skips: u32,
    session_minutes: f32,
    crashes: u8,
}

/// Acumulador por (usuário, mês)
#[derive(Default, Clone)]
struct Agg {
    plays: u64,
    skips: u64,
    session_sum: f64,
    session_n: u64,
    crashes: u64,
    events: u64,
}

impl Agg {
    fn push(&mut self, e: &LogEvent) {
        self.plays += e.plays as u64;
        self.skips += e.skips as u64;
        self.session_sum += e.session_minutes as f64;
        self.session_n += 1;
        self.crashes += e.crashes as u64;
        self.events += 1;
    }
}

fn main() -> io::Result<()> {
    // Lê eventos NDJSON de stdin (stream), escreve agregados NDJSON em stdout
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    let mut aggs: HashMap<(u64, u8), Agg> = HashMap::new();
    let mut rejected: u64 = 0;

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<LogEvent>(&line) {
            Ok(ev) => {
                if ev.user_id == 0 || !(1..=12).contains(&ev.mes) {
                    rejected += 1;
                    continue;
                }
                aggs.entry((ev.user_id, ev.mes)).or_default().push(&ev);
            }
            Err(_) => rejected += 1, // payload malformado: não derruba o serviço
        }
    }

    for ((user_id, mes), a) in aggs {
        let avg_minutes = a.session_sum / a.session_n.max(1) as f64;
        writeln!(
            out,
            "{{\"user_id\":{},\"mes\":{},\"plays\":{},\"skips\":{},\"session_minutes_avg\":{:.2},\"crashes\":{},\"log_events\":{}}}",
            user_id, mes, a.plays, a.skips, avg_minutes, a.crashes, a.events
        )?;
    }

    eprintln!("agregados emitidos | eventos malformados: {rejected}");
    Ok(())
}
