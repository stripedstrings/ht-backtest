"""Wall-clock stage timings written into every run artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StageTimings:
    parquet_load_s: float = 0.0
    primitives_s: float = 0.0
    gate_fsm_s: float = 0.0
    forward_tracker_s: float = 0.0
    reach_aggregation_s: float = 0.0
    assemble_tag_s: float = 0.0
    total_s: float = 0.0
    symbols: int = 0
    trades: int = 0
    notes: dict = field(default_factory=dict)

    def add(self, other: "StageTimings") -> "StageTimings":
        return StageTimings(
            parquet_load_s=self.parquet_load_s + other.parquet_load_s,
            primitives_s=self.primitives_s + other.primitives_s,
            gate_fsm_s=self.gate_fsm_s + other.gate_fsm_s,
            forward_tracker_s=self.forward_tracker_s + other.forward_tracker_s,
            reach_aggregation_s=self.reach_aggregation_s + other.reach_aggregation_s,
            assemble_tag_s=self.assemble_tag_s + other.assemble_tag_s,
            total_s=self.total_s + other.total_s,
            symbols=self.symbols + other.symbols,
            trades=self.trades + other.trades,
            notes={**self.notes, **other.notes},
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
