from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from src.ingestion.partition_writer import generate_telemetry_batch, write_partitioned_parquet


class DistributedAnalyticsEngine:
    """Motor OLAP out-of-core sobre arquivos Parquet particionados com pushdown de predicados."""

    def __init__(self, memory_limit: str = "1GB", threads: int = 4) -> None:
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute(f"SET memory_limit='{memory_limit}';")
        self.conn.execute(f"SET threads TO {threads};")

    def aggregate_carrier_performance(
        self,
        lakehouse_path: str | Path,
        filter_year: int | None = None,
    ) -> pl.DataFrame:
        """Executa query OLAP vetorizada lendo diretamente do particionamento Hive."""
        base_glob = f"{Path(lakehouse_path).as_posix()}/**/*.parquet"
        where_clause = f"WHERE event_year = {int(filter_year)}" if filter_year is not None else ""

        query = f"""
            SELECT
                region,
                carrier,
                COUNT(event_id) AS total_shipments,
                ROUND(AVG(freight_value), 2) AS avg_freight_value,
                ROUND(AVG(delivery_hours), 2) AS avg_delivery_hours,
                ROUND(SUM(CASE WHEN sla_breached THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS sla_breach_pct
            FROM read_parquet('{base_glob}', hive_partitioning = true)
            {where_clause}
            GROUP BY region, carrier
            ORDER BY region ASC, total_shipments DESC
        """
        arrow_result = self.conn.execute(query).fetch_arrow_table()
        return pl.from_arrow(arrow_result)

    def export_curated_mart(
        self,
        df: pl.DataFrame,
        output_parquet: str | Path = "data/lakehouse/curated_metrics/carrier_kpis.parquet",
    ) -> Path:
        """Materializa a tabela agregada Gold em Parquet de alta compressao (ZSTD)."""
        out_path = Path(output_parquet)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_path, compression="zstd")
        return out_path


def run_batch_analytics_job(
    raw_dir: str | Path = "data/lakehouse/raw_events",
    curated_file: str | Path = "data/lakehouse/curated_metrics/carrier_kpis.parquet",
    num_rows: int = 50_000,
) -> dict[str, Any]:
    """Orquestra a ingestao particionada e o processamento analitico out-of-core."""
    raw_df = generate_telemetry_batch(num_rows=num_rows)
    write_partitioned_parquet(raw_df, raw_dir)

    engine = DistributedAnalyticsEngine()
    kpis_df = engine.aggregate_carrier_performance(raw_dir, filter_year=2026)
    saved_path = engine.export_curated_mart(kpis_df, curated_file)

    return {
        "ingested_rows": num_rows,
        "aggregated_groups": kpis_df.height,
        "curated_path": str(saved_path),
    }


if __name__ == "__main__":
    summary = run_batch_analytics_job()
    print(f"Big Data Analytics Job finalizado: {summary}")