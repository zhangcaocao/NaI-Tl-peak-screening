#!/usr/bin/env python3
"""Launch the GUI or run the installation self-check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nai_peak.data import SpectrumDataset, default_example_paths
from nai_peak.model import PeakScreeningModel


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    default_pkl, default_annotations = default_example_paths(ROOT)
    parser = argparse.ArgumentParser(description="Interpretable NaI(Tl) peak screening")
    parser.add_argument("--dataset", type=Path, default=default_pkl)
    parser.add_argument("--annotations", type=Path, default=default_annotations)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--self-check", action="store_true", help="Run one CPU inference without opening the GUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        dataset = SpectrumDataset.load(args.dataset, args.annotations)
        record = dataset.record(0)
        model = PeakScreeningModel.from_pretrained(
            ROOT / "assets" / "model",
            backend="cpu",
            num_threads=args.threads,
            detector_profile=record.detector_profile,
        )
        result = model.predict_spectrum(record.data, return_contributions=True)
        print(json.dumps(result.to_summary(), indent=2))
        return 0

    from PySide6 import QtWidgets
    from nai_peak.gui import PeakScreeningWindow

    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Interpretable NaI(Tl) Peak Screening v1.0")
    window = PeakScreeningWindow(ROOT, args.dataset, args.annotations)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
