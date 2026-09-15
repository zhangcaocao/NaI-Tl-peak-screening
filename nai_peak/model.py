"""Interpretable local-shape NaI(Tl) peak-screening inference.

The module is intentionally self-contained.  It reproduces the published
preprocessing, permissive morphology, eleven interpretable local features,
XGBoost scoring, score-ranked resolution NMS, and regional thresholds.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import grey_opening
from scipy.signal import find_peaks
import xgboost as xgb


EPS = 1.0e-12
LN2 = math.log(2.0)
PARENT_HALF_WIDTH_FWHM = 3.0
MIN_RESOLVABLE_WIDTH_CHANNELS = 1.0
SIGMA_G = 1.0 / (2.0 * math.sqrt(2.0 * LN2))


@dataclass
class CandidateResult:
    candidate_id: str
    channel: float
    raw_channel: int
    expected_fwhm: float
    morphology_height: float
    morphology_prominence: float
    raw_mean_count_expected_fwhm: float
    count_filter_passed: bool
    features: dict[str, float]
    local_contributions: dict[str, float]
    contribution_bias: float | None
    score: float | None
    margin: float | None
    region: str
    threshold: float | None
    nms_allowed: bool | None
    suppressed_by: str | None
    retained: bool
    decision_reason: str


@dataclass
class InferenceResult:
    morphology_candidates: list[CandidateResult]
    scored_candidates: list[CandidateResult]
    peaks: list[CandidateResult]
    normalized_spectrum: np.ndarray
    morphology_opening: np.ndarray
    morphology_response: np.ndarray
    timing_ms: dict[str, float]
    runtime: dict[str, Any]

    def to_summary(self) -> dict[str, Any]:
        return {
            "morphology_candidates": len(self.morphology_candidates),
            "count_filter_passed": len(self.scored_candidates),
            "peaks": len(self.peaks),
            "timing_ms": self.timing_ms,
            "runtime": self.runtime,
            "peak_channels": [round(item.channel, 6) for item in self.peaks],
        }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_raw_spectrum(values: Iterable[float], channel_count: int) -> np.ndarray:
    spectrum = np.asarray(values, dtype=np.float64)
    if spectrum.ndim != 1 or spectrum.size != int(channel_count):
        raise ValueError(f"Expected one {channel_count}-channel spectrum")
    if not np.isfinite(spectrum).all():
        raise ValueError("Spectrum contains non-finite values")
    if np.any(spectrum < 0.0):
        raise ValueError("Spectrum contains negative counts")
    return spectrum


def _minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    low = float(np.min(array))
    high = float(np.max(array))
    span = high - low
    if span <= EPS * max(1.0, abs(low), abs(high)):
        return np.zeros_like(array)
    return np.clip((array - low) / span, 0.0, 1.0)


def _quadratic_refinement(response: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(response) - 1:
        return float(index)
    left, center, right = map(float, response[index - 1 : index + 2])
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator >= 0.0 or abs(denominator) < EPS:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index + np.clip(offset, -0.5, 0.5))


def _fwhm_at(fwhm: np.ndarray, channel: float) -> float:
    anchor = int(np.clip(np.rint(float(channel)), 0, len(fwhm) - 1))
    value = float(fwhm[anchor])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Invalid expected FWHM at channel {channel}")
    return value


def _candidate_generation_nms(records: list[dict], fwhm: np.ndarray, spacing: float) -> list[dict]:
    ordered = sorted(
        records,
        key=lambda item: (
            -float(item["morph_response"]),
            -float(item["morph_prominence"]),
            float(item["candidate_channel"]),
        ),
    )
    kept: list[dict] = []
    accepted_by_channel: list[tuple[float, int]] = []
    for record in ordered:
        channel = float(record["candidate_channel"])
        width = _fwhm_at(fwhm, channel)
        radius = float(spacing) * width
        lower = bisect_left(accepted_by_channel, (channel - radius, -1))
        upper = bisect_right(accepted_by_channel, (channel + radius, len(kept) + 1))
        conflict = False
        for _, accepted_index in accepted_by_channel[lower:upper]:
            accepted = kept[accepted_index]
            minimum_spacing = float(spacing) * min(width, accepted["expected_fwhm"])
            if abs(channel - accepted["candidate_channel"]) < minimum_spacing:
                conflict = True
                break
        if conflict:
            continue
        item = dict(record)
        item["expected_fwhm"] = width
        kept.append(item)
        insort(accepted_by_channel, (channel, len(kept) - 1))
    return sorted(kept, key=lambda item: item["candidate_channel"])


def _overlap_weighted_mean(spectrum: np.ndarray, center: float, width: float) -> float:
    left = max(-0.5, float(center) - 0.5 * float(width))
    right = min(len(spectrum) - 0.5, float(center) + 0.5 * float(width))
    if right <= left:
        return 0.0
    first = max(0, int(np.floor(left + 0.5)))
    last = min(len(spectrum) - 1, int(np.ceil(right - 0.5)))
    indices = np.arange(first, last + 1, dtype=int)
    overlap = np.maximum(
        np.minimum(indices + 0.5, right) - np.maximum(indices - 0.5, left), 0.0
    )
    denominator = float(overlap.sum())
    return 0.0 if denominator <= 0.0 else float(np.dot(spectrum[indices], overlap) / denominator)


def _mask_with_fallback(relative: np.ndarray, lower: float, upper: float) -> np.ndarray:
    mask = (relative >= lower) & (relative <= upper)
    if mask.any():
        return mask
    mask = np.zeros(len(relative), dtype=bool)
    mask[int(np.argmin(np.abs(relative - 0.5 * (lower + upper))))] = True
    return mask


def _linear_edge_background(relative: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    left = _mask_with_fallback(relative, -3.0, -2.5)
    right = _mask_with_fallback(relative, 2.5, 3.0)
    left_level, right_level = float(np.median(normalized[left])), float(np.median(normalized[right]))
    left_position, right_position = float(np.mean(relative[left])), float(np.mean(relative[right]))
    if abs(right_position - left_position) <= EPS:
        return np.full_like(normalized, 0.5 * (left_level + right_level))
    slope = (right_level - left_level) / (right_position - left_position)
    return left_level + slope * (relative - left_position)


def _local_maxima_features(relative: np.ndarray, normalized: np.ndarray) -> tuple[float, float]:
    maxima = find_peaks(normalized)[0]
    center_index = int(np.argmin(np.abs(relative)))
    other = maxima[maxima != center_index]
    nearest = min(PARENT_HALF_WIDTH_FWHM, float(np.min(np.abs(relative[other])))) if len(other) else PARENT_HALF_WIDTH_FWHM
    edge = np.abs(relative) >= 2.5
    baseline = float(np.median(normalized[edge])) if edge.any() else float(np.min(normalized))
    candidate_height = max(float(normalized[center_index]) - baseline, 0.0)
    competitors = [
        max(float(normalized[index]) - baseline, 0.0)
        for index in other
        if abs(float(relative[index])) > 0.5
    ]
    competitor = max(competitors) if competitors else 0.0
    ratio = candidate_height / (candidate_height + competitor + EPS)
    return nearest, float(np.clip(ratio, 0.0, 1.0))


def _shape_features(spectrum: np.ndarray, candidate: dict) -> dict[str, float]:
    center = float(candidate["candidate_channel"])
    effective_width = max(float(candidate["expected_fwhm"]), MIN_RESOLVABLE_WIDTH_CHANNELS)
    radius = int(np.ceil(PARENT_HALF_WIDTH_FWHM * effective_width))
    lower = max(0, int(np.floor(center)) - radius)
    upper = min(len(spectrum) - 1, int(np.ceil(center)) + radius)
    channels = np.arange(lower, upper + 1, dtype=int)
    relative = (channels.astype(float) - center) / effective_width
    keep = np.abs(relative) <= PARENT_HALF_WIDTH_FWHM
    channels, relative = channels[keep], relative[keep]
    if len(channels) < 3:
        raise ValueError("Candidate parent window has fewer than three channels")
    normalized = _minmax(spectrum[channels])
    background = _linear_edge_background(relative, normalized)
    excess = np.maximum(normalized - background, 0.0)

    def area(radius_fwhm: float) -> float:
        return float(np.sum(excess[np.abs(relative) <= radius_fwhm]))

    area_05, area_10, area_20 = area(0.5), area(1.0), area(2.0)
    left = float(np.sum(excess[(relative >= -1.0) & (relative < 0.0)]))
    right = float(np.sum(excess[(relative > 0.0) & (relative <= 1.0)]))
    central = np.abs(relative) <= 1.5
    mass = float(np.sum(excess[central]))
    centroid = float(np.sum(relative[central] * excess[central]) / mass) if mass > EPS else 0.0

    center_index = int(np.argmin(np.abs(relative)))
    if 0 < center_index < len(normalized) - 1:
        curvature = max(
            2.0 * float(normalized[center_index])
            - float(normalized[center_index - 1])
            - float(normalized[center_index + 1]),
            0.0,
        ) / 2.0
    else:
        curvature = 0.0

    core = np.abs(relative) <= 0.5
    edges = np.abs(relative) >= 2.5
    core_level = float(np.median(normalized[core])) if core.any() else 0.0
    edge_level = float(np.median(normalized[edges])) if edges.any() else 0.0
    edge_core_contrast = 0.5 * (np.clip(core_level - edge_level, -1.0, 1.0) + 1.0)

    differences = np.abs(np.diff(normalized))
    if len(differences):
        mid_relative = 0.5 * (relative[:-1] + relative[1:])
        side = np.abs(mid_relative) >= 1.5
        side_roughness = float(np.mean(differences[side])) if side.any() else 0.0
        total_roughness = float(np.mean(differences))
        roughness_ratio = side_roughness / (side_roughness + total_roughness + EPS)
    else:
        roughness_ratio = 0.0

    nearest, competitor_ratio = _local_maxima_features(relative, normalized)
    return {
        "morph_height": float(np.clip(candidate["morph_response"], 0.0, 1.0)),
        "morph_prominence": float(np.clip(candidate["morph_prominence"], 0.0, 1.0)),
        "core_concentration": area_05 / (area_10 + EPS),
        "inner_outer_ratio": area_10 / (area_20 + EPS),
        "left_right_symmetry": 1.0 - abs(left - right) / (left + right + EPS),
        "excess_centroid_offset": min(abs(centroid) / 1.5, 1.0),
        "normalized_curvature": float(np.clip(curvature, 0.0, 1.0)),
        "edge_core_contrast": float(np.clip(edge_core_contrast, 0.0, 1.0)),
        "local_roughness_ratio": float(np.clip(roughness_ratio, 0.0, 1.0)),
        "nearest_max_distance_fwhm": nearest,
        "competitor_height_ratio": competitor_ratio,
    }


class PeakScreeningModel:
    """Load and execute the multicore-CPU inference chain."""

    def __init__(self, model_dir: Path, card: dict[str, Any], detector_profile: str, num_threads: int):
        self.model_dir = Path(model_dir)
        self.card = card
        self.channel_count = int(card["channel_count"])
        self.feature_names = tuple(card["features"])
        self.feature_groups = dict(card["feature_groups"])
        if len(self.feature_names) != 11:
            raise RuntimeError(f"Expected the 11-feature contract, got {len(self.feature_names)}")
        self.num_threads = max(1, int(num_threads))
        self.detector_profile = str(detector_profile)
        if self.detector_profile not in card["detector_profiles"]:
            raise ValueError(f"Unknown detector profile: {self.detector_profile}")

        profile = card["detector_profiles"][self.detector_profile]
        fwhm_path = self.model_dir / profile["fwhm_file"]
        model_path = self.model_dir / card["model_file"]
        for path, expected in (
            (model_path, card["model_sha256"]),
            (fwhm_path, profile["fwhm_sha256"]),
        ):
            observed = _sha256_file(path)
            if observed != expected:
                raise RuntimeError(f"SHA-256 mismatch for {path.name}: {observed}")

        table = np.genfromtxt(fwhm_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        self.fwhm = np.asarray(table["expected_fwhm_channel"], dtype=np.float64)
        if self.fwhm.shape != (self.channel_count,) or not np.isfinite(self.fwhm).all() or np.any(self.fwhm <= 0.0):
            raise RuntimeError("FWHM response table is invalid")
        self.structure_width = int(profile["structure_width_channels"])
        self.booster = xgb.Booster(params={"nthread": self.num_threads})
        load_start = perf_counter()
        self.booster.load_model(model_path)
        self.model_load_ms = (perf_counter() - load_start) * 1000.0
        self._prediction_count = 0

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        backend: str = "cpu",
        num_threads: int | None = None,
        detector_profile: str = "external",
    ) -> "PeakScreeningModel":
        if str(backend).lower() != "cpu":
            raise ValueError("Version 1.0 supports the CPU backend only")
        directory = Path(model_dir).expanduser().resolve()
        card = json.loads((directory / "model_card.json").read_text(encoding="utf-8"))
        threads = int(num_threads or min(os.cpu_count() or 1, 8))
        return cls(directory, card, detector_profile, threads)

    def _generate_candidates(self, spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        normalized = _minmax(spectrum)
        opening = grey_opening(normalized, footprint=np.ones(self.structure_width, dtype=bool))
        response = np.maximum(normalized - opening, 0.0)
        morphology = self.card["morphology"]
        peaks, properties = find_peaks(
            response,
            height=float(morphology["height"]),
            prominence=float(morphology["prominence"]),
            distance=1,
        )
        heights = np.asarray(properties.get("peak_heights", np.zeros(len(peaks))), dtype=float)
        prominences = np.asarray(properties.get("prominences", np.zeros(len(peaks))), dtype=float)
        margin = int(morphology["edge_margin_channels"])
        records = []
        for peak, height, prominence in zip(peaks, heights, prominences):
            if peak < margin or peak >= self.channel_count - margin:
                continue
            records.append(
                {
                    "candidate_channel": _quadratic_refinement(response, int(peak)),
                    "candidate_channel_raw": int(peak),
                    "morph_response": float(height),
                    "morph_prominence": float(prominence),
                }
            )
        records = _candidate_generation_nms(records, self.fwhm, float(morphology["spacing_fwhm"]))
        for index, record in enumerate(records):
            record["candidate_id"] = f"candidate-{index:04d}"
        return normalized, opening, response, records

    def _classifier_nms(self, candidates: list[dict], scores: np.ndarray) -> tuple[np.ndarray, list[str | None]]:
        distance = float(self.card["classifier_nms_distance_fwhm"])
        allowed = np.zeros(len(candidates), dtype=bool)
        suppressed_by: list[str | None] = [None] * len(candidates)
        order = sorted(range(len(candidates)), key=lambda index: (-float(scores[index]), candidates[index]["candidate_id"]))
        kept: list[int] = []
        for index in order:
            suppressor = next(
                (
                    other
                    for other in kept
                    if abs(candidates[index]["candidate_channel"] - candidates[other]["candidate_channel"])
                    < distance * candidates[other]["expected_fwhm"]
                ),
                None,
            )
            if suppressor is None:
                allowed[index] = True
                kept.append(index)
            else:
                suppressed_by[index] = candidates[suppressor]["candidate_id"]
        return allowed, suppressed_by

    def predict_spectrum(self, raw_counts: Iterable[float], return_contributions: bool = True,
                         thresholds: dict[str, float] | None = None) -> InferenceResult:
        """Infer peaks; omitted cutoffs use the model-card default operating point."""
        if thresholds is None:
            default_key = self.card["default_operating_point"]
            thresholds = self.card["operating_points"][default_key]["thresholds"]
        thresholds = dict(thresholds)
        if set(thresholds) != {"low", "mid", "high"}:
            raise ValueError("Thresholds must contain low, mid, and high")
        thresholds = {region: float(value) for region, value in thresholds.items()}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds.values()):
            raise ValueError("Classifier thresholds must be finite values between 0 and 1")
        total_start = perf_counter()
        stage = perf_counter()
        spectrum = _validate_raw_spectrum(raw_counts, self.channel_count)
        validate_ms = (perf_counter() - stage) * 1000.0

        stage = perf_counter()
        normalized, opening, response, records = self._generate_candidates(spectrum)
        morphology_ms = (perf_counter() - stage) * 1000.0

        minimum_mean = float(self.card["candidate_count_filter"]["minimum_raw_mean_count_expected_fwhm"])
        eligible_indices = []
        for index, record in enumerate(records):
            mean = _overlap_weighted_mean(spectrum, record["candidate_channel"], record["expected_fwhm"])
            record["raw_mean_count_expected_fwhm"] = mean
            record["count_filter_passed"] = mean >= minimum_mean
            if record["count_filter_passed"]:
                eligible_indices.append(index)

        stage = perf_counter()
        eligible_records = [records[index] for index in eligible_indices]
        if self.num_threads > 1 and len(eligible_records) >= 16:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                feature_rows = list(executor.map(lambda candidate: _shape_features(spectrum, candidate), eligible_records))
        else:
            feature_rows = [_shape_features(spectrum, candidate) for candidate in eligible_records]
        feature_ms = (perf_counter() - stage) * 1000.0

        stage = perf_counter()
        if feature_rows:
            extracted_names = set(feature_rows[0])
            if extracted_names != set(self.feature_names):
                missing = sorted(set(self.feature_names) - extracted_names)
                unexpected = sorted(extracted_names - set(self.feature_names))
                raise RuntimeError(
                    f"Feature extractor/model-card mismatch; missing={missing}, unexpected={unexpected}"
                )
            matrix = np.asarray([[row[name] for name in self.feature_names] for row in feature_rows], dtype=np.float32)
            dmatrix = xgb.DMatrix(matrix)
            scores = np.asarray(self.booster.predict(dmatrix), dtype=float)
            margins = np.asarray(self.booster.predict(dmatrix, output_margin=True), dtype=float)
            contributions = (
                np.asarray(self.booster.predict(dmatrix, pred_contribs=True), dtype=float)
                if return_contributions
                else np.zeros((len(matrix), len(self.feature_names) + 1), dtype=float)
            )
            contribution_tolerance = float(self.card.get("contribution_reconstruction_tolerance", 5.0e-5))
            if return_contributions and not np.allclose(
                contributions.sum(axis=1), margins, rtol=0.0, atol=contribution_tolerance
            ):
                maximum_error = float(np.max(np.abs(contributions.sum(axis=1) - margins)))
                raise RuntimeError(
                    "Feature contributions do not reconstruct the XGBoost margin; "
                    f"maximum absolute error is {maximum_error:.3e}"
                )
        else:
            scores = np.empty(0, dtype=float)
            margins = np.empty(0, dtype=float)
            contributions = np.empty((0, len(self.feature_names) + 1), dtype=float)
        classifier_ms = (perf_counter() - stage) * 1000.0

        stage = perf_counter()
        eligible_for_nms = [records[index] for index in eligible_indices]
        nms_allowed, suppressed_by = self._classifier_nms(eligible_for_nms, scores) if len(scores) else (np.empty(0, dtype=bool), [])
        feature_by_index = {source_index: row_index for row_index, source_index in enumerate(eligible_indices)}
        candidates: list[CandidateResult] = []
        for source_index, record in enumerate(records):
            if source_index not in feature_by_index:
                candidates.append(
                    CandidateResult(
                        candidate_id=record["candidate_id"], channel=record["candidate_channel"], raw_channel=record["candidate_channel_raw"],
                        expected_fwhm=record["expected_fwhm"], morphology_height=record["morph_response"], morphology_prominence=record["morph_prominence"],
                        raw_mean_count_expected_fwhm=record["raw_mean_count_expected_fwhm"], count_filter_passed=False,
                        features={}, local_contributions={}, contribution_bias=None, score=None, margin=None,
                        region=self._region(record["candidate_channel"]), threshold=None, nms_allowed=None, suppressed_by=None,
                        retained=False, decision_reason="Below candidate count filter",
                    )
                )
                continue
            row_index = feature_by_index[source_index]
            region = self._region(record["candidate_channel"])
            threshold = float(thresholds[region])
            score = float(scores[row_index])
            allowed = bool(nms_allowed[row_index])
            retained = allowed and score >= threshold
            if not allowed:
                nms_distance = float(self.card["classifier_nms_distance_fwhm"])
                reason = f"Suppressed by {suppressed_by[row_index]} at {nms_distance:.2f} FWHM NMS"
            elif score < threshold:
                reason = f"Score below {region}-region threshold"
            else:
                reason = "Retained peak"
            local = (
                {name: float(contributions[row_index, index]) for index, name in enumerate(self.feature_names)}
                if return_contributions else {}
            )
            candidates.append(
                CandidateResult(
                    candidate_id=record["candidate_id"], channel=record["candidate_channel"], raw_channel=record["candidate_channel_raw"],
                    expected_fwhm=record["expected_fwhm"], morphology_height=record["morph_response"], morphology_prominence=record["morph_prominence"],
                    raw_mean_count_expected_fwhm=record["raw_mean_count_expected_fwhm"], count_filter_passed=True,
                    features=feature_rows[row_index], local_contributions=local,
                    contribution_bias=(float(contributions[row_index, -1]) if return_contributions else None),
                    score=score, margin=float(margins[row_index]), region=region, threshold=threshold,
                    nms_allowed=allowed, suppressed_by=suppressed_by[row_index], retained=retained, decision_reason=reason,
                )
            )
        nms_threshold_ms = (perf_counter() - stage) * 1000.0

        self._prediction_count += 1
        total_ms = (perf_counter() - total_start) * 1000.0
        timing = {
            "input_validation": validate_ms,
            "normalization_and_morphology": morphology_ms,
            "feature_extraction": feature_ms,
            "classifier": classifier_ms,
            "nms_and_thresholds": nms_threshold_ms,
            "total": total_ms,
            "model_load_separate": self.model_load_ms,
        }
        runtime = {
            "thresholds": dict(thresholds),
            "backend": "cpu",
            "num_threads": self.num_threads,
            "model_version": str(self.card["version"]),
            "feature_count": len(self.feature_names),
            "detector_profile": self.detector_profile,
            "structure_width_channels": self.structure_width,
            "xgboost_version": xgb.__version__,
            "run_kind": "cold" if self._prediction_count == 1 else "warm",
        }
        return InferenceResult(
            morphology_candidates=candidates,
            scored_candidates=[candidate for candidate in candidates if candidate.count_filter_passed],
            peaks=[candidate for candidate in candidates if candidate.retained],
            normalized_spectrum=normalized,
            morphology_opening=opening,
            morphology_response=response,
            timing_ms=timing,
            runtime=runtime,
        )

    @staticmethod
    def _region(channel: float) -> str:
        return "low" if channel < 100.0 else ("mid" if channel < 400.0 else "high")

    def predict_many(self, spectra: Iterable[Iterable[float]], return_contributions: bool = False) -> list[InferenceResult]:
        """Deterministic batch convenience API using the configured CPU runtime."""
        return [self.predict_spectrum(spectrum, return_contributions=return_contributions) for spectrum in spectra]
