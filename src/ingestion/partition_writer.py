from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from pydantic import BaseModel, ConfigDict, Field


class PartitionConfig(BaseModel):
    """Configuracao estrita para geracao e particionamento Hive-style em Parquet."""

    model_config = ConfigDict(extra="forbid")

    num_rows: int = Field(default=50_000, gt=100)
    base_dir: Path = Field(default=Path("data/lakehouse/raw_events"))


def generate_telemetry_batch(num_rows: int = 50_000) -> pl.DataFrame:
    """Gera lote massivo de eventos logisticos vetorizados via Polars."""
    regions = ["SE", "SUL", "NE", "CO", "N"]
    carriers = ["FAST_LOG", "TRANS_BR", "AERO_CARGO", "ECO_EXPRESS"]

    return pl.DataFrame(
        {
            "event_id": range(1, num_rows + 1),
            "region": [regions[i % len(regions)] for i in range(num_rows)],
            "carrier": [carriers[i % len(carriers)] for i in range(num_rows)],
            "event_year": [2025 if i % 3 == 0 else 2026 for i in range(num_rows)],
            "freight_value": [round(45.0 + (i % 250) * 1.75, 2) for i in range(num_rows)],
            "delivery_hours": [12 + (i % 96) for i in range(num_rows)],
            "sla_breached": [(i % 96) > 72 for i in range(num_rows)],
        }
    )


def write_partitioned_parquet(df: pl.DataFrame, output_dir: str | Path) -> Path:
    """Escreve DataFrame em formato Apache Parquet particionado (Hive-style) via PyArrow."""
    target_path = Path(output_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    arrow_table: pa.Table = df.to_arrow()
    ds.write_dataset(
        data=arrow_table,
        base_dir=str(target_path),
        format="parquet",
        partitioning=["event_year", "region"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
    )
    return target_path