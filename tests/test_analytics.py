from __future__ import annotations

from pathlib import Path

from src.engine.analytics import DistributedAnalyticsEngine, run_batch_analytics_job
from src.ingestion.partition_writer import generate_telemetry_batch, write_partitioned_parquet


def test_hive_partitioned_parquet_generation(tmp_path: Path) -> None:
    """Valida a escrita colunar particionada no padrao Hive (event_year=.../region=...)."""
    df = generate_telemetry_batch(num_rows=1_000)
    out_dir = write_partitioned_parquet(df, tmp_path / "raw_events")

    partitions = list(out_dir.rglob("*.parquet"))
    assert len(partitions) > 0
    assert any("event_year=" in str(p) for p in partitions)
    assert any("region=" in str(p) for p in partitions)


def test_duckdb_predicate_pushdown_and_aggregation(tmp_path: Path) -> None:
    """Valida a agregacao OLAP out-of-core e filtro de particao via DuckDB + PyArrow."""
    df = generate_telemetry_batch(num_rows=2_500)
    lake_dir = write_partitioned_parquet(df, tmp_path / "lake")

    engine = DistributedAnalyticsEngine(memory_limit="512MB", threads=2)
    res_2026 = engine.aggregate_carrier_performance(lake_dir, filter_year=2026)

    assert res_2026.height > 0
    assert "total_shipments" in res_2026.columns
    assert "sla_breach_pct" in res_2026.columns
    assert res_2026["total_shipments"].sum() < 2_500


def test_end_to_end_batch_job(tmp_path: Path) -> None:
    """Valida a execucao completa do job de ingestao e geracao de Data Mart ZSTD."""
    raw_dir = tmp_path / "raw"
    curated_file = tmp_path / "curated" / "kpis.parquet"

    metrics = run_batch_analytics_job(raw_dir=raw_dir, curated_file=curated_file, num_rows=3_000)
    assert metrics["ingested_rows"] == 3_000
    assert metrics["aggregated_groups"] == 20
    assert curated_file.exists()