"""Dataset adapter for the bundled demonstration spectra and user data."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "spectrum_id",
    "dataset",
    "data",
    "nuclides",
    "gross_counts",
    "snr",
    "detector_profile",
}


@dataclass(frozen=True)
class SpectrumRecord:
    spectrum_id: int
    dataset: str
    data: np.ndarray
    nuclides: tuple[str, ...]
    gross_counts: float
    snr: float
    detector_profile: str
    source_spectrum_id: int | None
    annotation: dict[str, Any] | None


class SpectrumDataset:
    """Validated spectrum table with optional Gaussian-fit annotations."""

    def __init__(self, frame: pd.DataFrame, annotations: dict[str, Any] | None = None):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        work = frame.reset_index(drop=True).copy()
        if work.spectrum_id.duplicated().any():
            raise ValueError("spectrum_id values must be unique")
        expected = list(range(len(work)))
        if work.spectrum_id.astype(int).tolist() != expected:
            raise ValueError("spectrum_id must be contiguous and ordered from zero")
        checked = []
        for spectrum in work.data:
            values = np.asarray(spectrum, dtype=np.float64)
            if values.shape != (1024,) or not np.isfinite(values).all() or np.any(values < 0.0):
                raise ValueError("Every spectrum must contain 1024 finite non-negative counts")
            checked.append(values)
        work["data"] = checked
        if not work.detector_profile.astype(str).isin(["internal", "external"]).all():
            raise ValueError("detector_profile must be internal or external")
        self.frame = work
        self.annotations = annotations
        if annotations is not None:
            if int(annotations.get("n_spectra", -1)) != len(work):
                raise ValueError("Annotation and spectrum counts differ")
            if int(annotations.get("n_channels", -1)) != 1024:
                raise ValueError("Annotation channel count is not 1024")
            keys = {int(key) for key in annotations.get("spectra", {})}
            if keys != set(expected):
                raise ValueError("Annotation spectrum IDs do not match the pkl file")

    @classmethod
    def load(cls, pkl_path: str | Path, annotation_path: str | Path | None = None) -> "SpectrumDataset":
        path = Path(pkl_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_pickle(path)
        annotations = None
        if annotation_path:
            annotation_file = Path(annotation_path).expanduser().resolve()
            annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
        return cls(frame, annotations)

    @property
    def datasets(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.frame.dataset.astype(str)))

    def ids_for_dataset(self, dataset: str) -> list[int]:
        return self.frame.loc[self.frame.dataset.astype(str).eq(str(dataset)), "spectrum_id"].astype(int).tolist()

    def record(self, spectrum_id: int) -> SpectrumRecord:
        row = self.frame.iloc[int(spectrum_id)]
        value = row.nuclides
        if isinstance(value, (tuple, list, np.ndarray)):
            nuclides = tuple(str(item) for item in value)
        else:
            nuclides = (str(value),)
        annotation = None
        if self.annotations is not None:
            annotation = self.annotations["spectra"].get(str(int(spectrum_id)))
        source_id = row.get("source_spectrum_id", None)
        return SpectrumRecord(
            spectrum_id=int(row.spectrum_id),
            dataset=str(row.dataset),
            data=np.asarray(row.data, dtype=np.float64),
            nuclides=nuclides,
            gross_counts=float(row.gross_counts),
            snr=float(row.snr),
            detector_profile=str(row.detector_profile),
            source_spectrum_id=(None if source_id is None or pd.isna(source_id) else int(source_id)),
            annotation=annotation,
        )

    def __len__(self) -> int:
        return len(self.frame)


def default_example_paths(project_root: str | Path) -> tuple[Path, Path]:
    root = Path(project_root).expanduser().resolve()
    examples = root / "assets" / "examples"
    return examples / "nai_demo_500.pkl", examples / "nai_demo_annotations.json"
