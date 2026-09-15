"""English desktop GUI for the NaI(Tl) peak-screening model."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Callable

import numpy as np
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .data import SpectrumDataset
from .model import CandidateResult, InferenceResult, PeakScreeningModel


COLORS = {
    "spectrum": "#1F2937",
    "response": "#0F9D8A",
    "morphology": "#7A7A7A",
    "model": "#0072B2",
    "truth": "#D55E00",
}


class PeakScreeningWindow(QtWidgets.QMainWindow):
    def __init__(self, project_root: Path, dataset_path: Path, annotation_path: Path | None):
        super().__init__()
        self.project_root = Path(project_root)
        self.model_dir = self.project_root / "assets" / "model"
        self.model_card = json.loads((self.model_dir / "model_card.json").read_text())
        self.dataset_path = Path(dataset_path)
        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.dataset = SpectrumDataset.load(self.dataset_path, self.annotation_path)
        self.model: PeakScreeningModel | None = None
        self.model_key: tuple[str, int] | None = None
        self.current_record = None
        self.current_result: InferenceResult | None = None
        self.candidate_by_id: dict[str, CandidateResult] = {}
        self._build_ui()
        self._populate_datasets()
        self._select_first_sample()

    def _build_ui(self) -> None:
        self.setWindowTitle("Interpretable NaI(Tl) Peak Screening v1.0")
        self.resize(1500, 900)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        self.open_button = QtWidgets.QPushButton("Open dataset")
        self.open_button.clicked.connect(self._open_dataset)
        controls.addWidget(self.open_button)
        controls.addWidget(QtWidgets.QLabel("Data subset"))
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._dataset_changed)
        controls.addWidget(self.dataset_combo)
        controls.addWidget(QtWidgets.QLabel("Spectrum"))
        self.sample_combo = QtWidgets.QComboBox()
        self.sample_combo.currentIndexChanged.connect(self._sample_changed)
        controls.addWidget(self.sample_combo, 1)
        self.previous_button = QtWidgets.QPushButton("Previous spectrum")
        self.next_button = QtWidgets.QPushButton("Next spectrum")
        self.previous_button.clicked.connect(lambda: self._move_sample(-1))
        self.next_button.clicked.connect(lambda: self._move_sample(1))
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)
        controls.addWidget(QtWidgets.QLabel("CPU threads"))
        self.thread_spin = QtWidgets.QSpinBox()
        self.thread_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.thread_spin.setValue(min(8, max(1, os.cpu_count() or 1)))
        controls.addWidget(self.thread_spin)
        self.run_button = QtWidgets.QPushButton("Run inference")
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run_inference)
        controls.addWidget(self.run_button)
        layout.addLayout(controls)

        threshold_controls = QtWidgets.QHBoxLayout()
        threshold_controls.addWidget(QtWidgets.QLabel("Classifier thresholds"))
        self.threshold_preset = QtWidgets.QComboBox()
        for key in ("fp_1", "best_f1", "fp_3", "fp_5"):
            self.threshold_preset.addItem(self.model_card["operating_points"][key]["label"], key)
        self.threshold_preset.addItem("Custom", "custom")
        threshold_controls.addWidget(self.threshold_preset)
        self.threshold_spins = {}
        default_key = self.model_card["default_operating_point"]
        defaults = self.model_card["operating_points"][default_key]["thresholds"]
        for region, label in [("low", "Channel < 100"), ("mid", "100 ≤ channel < 400"), ("high", "Channel ≥ 400")]:
            threshold_controls.addWidget(QtWidgets.QLabel(label))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setDecimals(10)
            spin.setSingleStep(0.01)
            spin.setValue(defaults[region])
            spin.setToolTip("Higher thresholds retain fewer candidates. Click Run inference to apply.")
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._threshold_changed)
            self.threshold_spins[region] = spin
            threshold_controls.addWidget(spin)
        self.reset_thresholds_button = QtWidgets.QPushButton("Reset to 1 FP/spectrum")
        self.reset_thresholds_button.clicked.connect(lambda: self._select_threshold_preset(0))
        threshold_controls.addWidget(self.reset_thresholds_button)
        self.threshold_preset.currentIndexChanged.connect(self._select_threshold_preset)
        layout.addLayout(threshold_controls)
        self.preset_summary = QtWidgets.QLabel()
        layout.addWidget(self.preset_summary)
        self._update_preset_summary(default_key)

        toggles = QtWidgets.QHBoxLayout()
        self.show_response = self._checkbox("Morphology response (scaled)", False, self._refresh_plot)
        self.show_morphology = self._checkbox("Morphology candidates", True, self._refresh_plot)
        self.show_model = self._checkbox("Model peaks", True, self._refresh_plot)
        self.show_truth = self._checkbox("Fitted references", True, self._refresh_plot)
        for widget in (self.show_response, self.show_morphology, self.show_model, self.show_truth):
            toggles.addWidget(widget)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "Channel")
        self.plot.setLabel("left", "Counts")
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.addLegend(offset=(10, 10))
        splitter.addWidget(self.plot)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        self.sample_info = QtWidgets.QLabel()
        self.sample_info.setWordWrap(True)
        right_layout.addWidget(self.sample_info)
        right_layout.addWidget(QtWidgets.QLabel("Processing time"))
        self.runtime_text = QtWidgets.QPlainTextEdit()
        self.runtime_text.setReadOnly(True)
        self.runtime_text.setMaximumHeight(120)
        right_layout.addWidget(self.runtime_text)

        right_layout.addWidget(QtWidgets.QLabel("Candidates"))
        self.candidate_table = QtWidgets.QTableWidget(0, 5)
        self.candidate_table.setHorizontalHeaderLabels(["ID", "Channel", "Score", "Status", "Decision reason"])
        self.candidate_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.itemSelectionChanged.connect(self._candidate_table_selection)
        right_layout.addWidget(self.candidate_table, 2)

        right_layout.addWidget(QtWidgets.QLabel("Candidate evidence"))
        self.feature_table = QtWidgets.QTableWidget(0, 4)
        self.feature_table.setHorizontalHeaderLabels(["Evidence group", "Feature", "Feature value", "Local contribution"])
        feature_header = self.feature_table.horizontalHeader()
        feature_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        feature_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        feature_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        feature_header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        right_layout.addWidget(self.feature_table, 2)
        self.candidate_summary = QtWidgets.QLabel("Select a candidate to inspect its evidence.")
        self.candidate_summary.setWordWrap(True)
        self.candidate_summary.setMinimumHeight(44)
        right_layout.addWidget(self.candidate_summary)
        splitter.addWidget(right)
        splitter.setSizes([940, 560])
        self.statusBar().showMessage("Ready")

    @staticmethod
    def _checkbox(text: str, checked: bool, callback: Callable) -> QtWidgets.QCheckBox:
        box = QtWidgets.QCheckBox(text)
        box.setChecked(checked)
        box.toggled.connect(callback)
        return box

    def _threshold_changed(self, *_args) -> None:
        self.threshold_preset.blockSignals(True)
        self.threshold_preset.setCurrentIndex(4)
        self.threshold_preset.blockSignals(False)
        self._update_preset_summary("custom")
        if self.current_record is not None:
            self._load_current_sample()
        self.statusBar().showMessage("Thresholds changed. Click Run inference to apply.")

    def _select_threshold_preset(self, index: int) -> None:
        key = self.threshold_preset.itemData(index)
        if key == "custom":
            return
        values = self.model_card["operating_points"][key]["thresholds"]
        for region, spin in self.threshold_spins.items():
            spin.blockSignals(True)
            spin.setValue(values[region])
            spin.blockSignals(False)
        self._threshold_changed()
        self.threshold_preset.blockSignals(True)
        self.threshold_preset.setCurrentIndex(index)
        self.threshold_preset.blockSignals(False)
        self._update_preset_summary(key)

    def _update_preset_summary(self, key: str) -> None:
        if key == "custom":
            self.preset_summary.setText(
                "Custom thresholds. Click Run inference to apply; performance depends on the input spectra."
            )
            return
        point = self.model_card["operating_points"][key]
        threshold = point["thresholds"]["mid"]
        self.preset_summary.setText(
            f"Validation operating point: threshold {threshold:.10f}, "
            f"Recall {100 * point['validation_recall']:.2f}%, "
            f"FP/spectrum {point['validation_fp_per_spectrum']:.3f}. "
            "Performance on new spectra may differ."
        )

    def _populate_datasets(self) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        self.dataset_combo.addItems(self.dataset.datasets)
        self.dataset_combo.blockSignals(False)

    def _select_first_sample(self) -> None:
        if self.dataset_combo.count():
            self.dataset_combo.setCurrentIndex(0)
            self._dataset_changed(self.dataset_combo.currentText())

    def _dataset_changed(self, dataset: str) -> None:
        ids = self.dataset.ids_for_dataset(dataset)
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        for spectrum_id in ids:
            self.sample_combo.addItem(f"{dataset} | spectrum {spectrum_id}", spectrum_id)
        self.sample_combo.blockSignals(False)
        if ids:
            self.sample_combo.setCurrentIndex(0)
            self._load_current_sample()

    def _sample_changed(self, _: int) -> None:
        if self.sample_combo.currentIndex() >= 0:
            self._load_current_sample()

    def _move_sample(self, offset: int) -> None:
        count = self.sample_combo.count()
        if count:
            self.sample_combo.setCurrentIndex((self.sample_combo.currentIndex() + offset) % count)

    def _load_current_sample(self) -> None:
        spectrum_id = int(self.sample_combo.currentData())
        self.current_record = self.dataset.record(spectrum_id)
        self.current_result = None
        self.candidate_by_id.clear()
        self.candidate_table.setRowCount(0)
        self.feature_table.setRowCount(0)
        self.runtime_text.clear()
        self.candidate_summary.setText("Select a candidate to inspect its evidence.")
        self.sample_info.setText(
            f"<b>Dataset:</b> {self.current_record.dataset}<br>"
            f"<b>Spectrum ID:</b> {self.current_record.spectrum_id} &nbsp; "
            f"<b>Source ID:</b> {self.current_record.source_spectrum_id}<br>"
            f"<b>Nuclide labels:</b> {', '.join(self.current_record.nuclides)}<br>"
            f"<b>Gross counts:</b> {self.current_record.gross_counts:.0f} &nbsp; "
            f"<b>SNR:</b> {self.current_record.snr:.3f} &nbsp; "
            f"<b>Detector profile:</b> {self.current_record.detector_profile}"
        )
        self._refresh_plot()
        self.statusBar().showMessage("Spectrum loaded. Run inference to generate candidates.")

    def _open_dataset(self) -> None:
        pkl_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open spectrum dataset", str(self.dataset_path.parent), "Pickle files (*.pkl)")
        if not pkl_path:
            return
        annotation_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open annotation project (optional)", str(Path(pkl_path).parent), "JSON files (*.json)")
        try:
            self.dataset_path = Path(pkl_path)
            self.annotation_path = Path(annotation_path) if annotation_path else None
            self.dataset = SpectrumDataset.load(self.dataset_path, self.annotation_path)
            self._populate_datasets()
            self._select_first_sample()
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Cannot open dataset", str(error))

    def run_inference(self) -> None:
        if self.current_record is None:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            key = (self.current_record.detector_profile, int(self.thread_spin.value()))
            if self.model is None or self.model_key != key:
                self.model = PeakScreeningModel.from_pretrained(
                    self.model_dir,
                    backend="cpu",
                    num_threads=key[1],
                    detector_profile=key[0],
                )
                self.model_key = key
            preset_key = self.threshold_preset.currentData()
            thresholds = ({region: spin.value() for region, spin in self.threshold_spins.items()}
                          if preset_key == "custom" else
                          self.model_card["operating_points"][preset_key]["thresholds"])
            self.current_result = self.model.predict_spectrum(
                self.current_record.data, return_contributions=True,
                thresholds=thresholds,
            )
            self.candidate_by_id = {candidate.candidate_id: candidate for candidate in self.current_result.morphology_candidates}
            self._populate_candidates()
            self._refresh_plot()
            timing = self.current_result.timing_ms
            runtime = self.current_result.runtime
            self.runtime_text.setPlainText(
                f"Model: v{runtime['model_version']} | Features: {runtime['feature_count']} "
                f"| Local window: +/-3 FWHM\n"
                f"Run: {runtime['run_kind']} | Backend: CPU | Threads: {runtime['num_threads']}\n"
                f"Thresholds (low/mid/high): "
                f"{runtime['thresholds']['low']:.6f} / {runtime['thresholds']['mid']:.6f} / {runtime['thresholds']['high']:.6f}\n"
                f"Total: {timing['total']:.2f} ms\n"
                f"Morphology: {timing['normalization_and_morphology']:.2f} ms | "
                f"Features: {timing['feature_extraction']:.2f} ms | "
                f"Classifier: {timing['classifier']:.2f} ms | "
                f"NMS + thresholds: {timing['nms_and_thresholds']:.2f} ms\n"
                f"Model load (reported separately): {timing['model_load_separate']:.2f} ms"
            )
            self.statusBar().showMessage(
                f"Inference complete: {len(self.current_result.morphology_candidates)} morphology candidates, "
                f"{len(self.current_result.peaks)} model peaks"
            )
        except Exception as error:
            QtWidgets.QMessageBox.critical(self, "Inference failed", str(error))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _populate_candidates(self) -> None:
        candidates = self.current_result.morphology_candidates if self.current_result else []
        self.candidate_table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            status = "Peak" if candidate.retained else ("Passed count filter" if candidate.count_filter_passed else "Below count filter")
            values = [
                candidate.candidate_id,
                f"{candidate.channel:.3f}",
                "—" if candidate.score is None else f"{candidate.score:.5f}",
                status,
                candidate.decision_reason,
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, candidate.candidate_id)
                self.candidate_table.setItem(row, column, item)

    def _refresh_plot(self) -> None:
        self.plot.clear()
        if self.current_record is None:
            return
        spectrum = self.current_record.data
        channels = np.arange(len(spectrum))
        self.plot.plot(channels, spectrum, pen=pg.mkPen(COLORS["spectrum"], width=1.2), name="Spectrum")
        if self.current_result is not None and self.show_response.isChecked():
            scaled = self.current_result.morphology_response * max(float(np.max(spectrum)), 1.0)
            self.plot.plot(channels, scaled, pen=pg.mkPen(COLORS["response"], width=1.0, style=QtCore.Qt.PenStyle.DashLine), name="Morphology response (scaled)")
        if self.current_result is not None and self.show_morphology.isChecked():
            candidates = self.current_result.morphology_candidates
            self._add_scatter(candidates, spectrum, COLORS["morphology"], "o", 6, "Morphology candidates")
        if self.current_result is not None and self.show_model.isChecked():
            self._add_scatter(self.current_result.peaks, spectrum, COLORS["model"], "t", 11, "Model peaks")
        if self.show_truth.isChecked():
            centers = self._reference_centers()
            if centers:
                y = [spectrum[int(np.clip(np.rint(center), 0, len(spectrum) - 1))] for center, _ in centers]
                truth = pg.ScatterPlotItem(
                    [center for center, _ in centers], y,
                    pen=pg.mkPen(COLORS["truth"]), brush=pg.mkBrush(COLORS["truth"]), symbol="d", size=9,
                    name="Fitted references",
                )
                self.plot.addItem(truth)

    def _add_scatter(self, candidates: list[CandidateResult], spectrum: np.ndarray, color: str, symbol: str, size: int, name: str) -> None:
        if not candidates:
            return
        spots = []
        for candidate in candidates:
            anchor = int(np.clip(np.rint(candidate.channel), 0, len(spectrum) - 1))
            spots.append({"pos": (candidate.channel, float(spectrum[anchor])), "data": candidate.candidate_id})
        item = pg.ScatterPlotItem(spots=spots, pen=pg.mkPen(color), brush=pg.mkBrush(color), symbol=symbol, size=size, name=name)
        item.sigClicked.connect(self._scatter_clicked)
        self.plot.addItem(item)

    def _reference_centers(self) -> list[tuple[float, str]]:
        if self.current_record is None or self.current_record.annotation is None:
            return []
        centers = []
        for peak in self.current_record.annotation.get("peaks", []):
            for component in peak.get("fit_peak_params") or []:
                center = component.get("center")
                if center is not None:
                    centers.append((float(center), str(peak.get("assigned_nuclide", ""))))
        return centers

    def _scatter_clicked(self, _item, points) -> None:
        if points:
            self._show_candidate(str(points[0].data()))

    def _candidate_table_selection(self) -> None:
        items = self.candidate_table.selectedItems()
        if items:
            self._show_candidate(str(items[0].data(QtCore.Qt.ItemDataRole.UserRole)))

    def _show_candidate(self, candidate_id: str) -> None:
        candidate = self.candidate_by_id.get(candidate_id)
        if candidate is None:
            return
        for row in range(self.candidate_table.rowCount()):
            item = self.candidate_table.item(row, 0)
            if item and item.text() == candidate_id:
                self.candidate_table.selectRow(row)
                break
        group_by_feature = {
            feature: group
            for group, features in (self.model.feature_groups.items() if self.model else [])
            for feature in features
        }
        names = list(self.model.feature_names) if self.model else []
        self.feature_table.setRowCount(len(names))
        for row, feature in enumerate(names):
            values = [
                group_by_feature.get(feature, ""), feature,
                f"{candidate.features.get(feature, float('nan')):.6f}",
                f"{candidate.local_contributions.get(feature, float('nan')):+.6f}",
            ]
            for column, value in enumerate(values):
                self.feature_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        score = "not scored" if candidate.score is None else f"{candidate.score:.6f}"
        threshold = "not applicable" if candidate.threshold is None else f"{candidate.threshold:.6f}"
        self.candidate_summary.setText(
            f"<b>{candidate.candidate_id}</b> at channel {candidate.channel:.3f}; "
            f"FWHM geometry scale {candidate.expected_fwhm:.3f}. "
            f"Score: {score}; threshold: {threshold}. {candidate.decision_reason}. "
            f"Raw-window mean count: {candidate.raw_mean_count_expected_fwhm:.3f}."
        )
