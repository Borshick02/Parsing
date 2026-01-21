from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .context import PipelineContext
from .handlers import (
    FinalizeFeaturesHandler,
    HhPreprocessHandler,
    LoadCsvChunksHandler,
    SaveArtifactsHandler,
)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    target_column: str = "ЗП"
    chunksize: int = 100_000
    drop_duplicates: bool = True
    scale_age_minmax: bool = True


def build_pipeline(config: PipelineConfig):
    loader = LoadCsvChunksHandler(chunksize=config.chunksize)
    processor = HhPreprocessHandler(target_column=config.target_column)
    finalize = FinalizeFeaturesHandler(
        drop_duplicates=config.drop_duplicates,
        scale_age_minmax=config.scale_age_minmax,
    )
    saver = SaveArtifactsHandler()

    loader.set_next(processor).set_next(finalize).set_next(saver)
    return loader


def run_pipeline(csv_path: Path, *, config: PipelineConfig) -> PipelineContext:
    ctx = PipelineContext(
        input_path=csv_path,
        output_dir=csv_path.parent,
    )
    chain = build_pipeline(config)
    return chain.handle(ctx)
