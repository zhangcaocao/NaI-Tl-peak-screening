# Interpretable NaI(Tl) peak screening

This repository provides the inference implementation and graphical interface
for *Interpretable local-shape screening for morphology-based peak detection in
NaI(Tl) gamma-ray spectra*.

![GUI](GUI.png)
Repository: <https://github.com/zhangcaocao/NaI-Tl-peak-screening>

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python run_gui.py
```

## Graphical interface

The GUI opens the bundled 500-spectrum demonstration set by default.
It displays the count spectrum, morphological candidates, screened peaks,
Gaussian-fit references, nuclide labels, candidate scores, the eleven local
shape measurements, signed model contributions, and processing time.


## Python API

```python
from nai_peak import PeakScreeningModel

model = PeakScreeningModel.from_pretrained(
    "assets/model",
    backend="cpu",
    num_threads=8,
    detector_profile="external",
)
result = model.predict_spectrum(raw_counts, return_contributions=True)
print(result.to_summary())
```

The input is one finite, non-negative, 1024-channel count spectrum.
`result.morphology_candidates` contains the permissive candidate set,
`result.scored_candidates` contains candidates that pass the count filter, and
`result.peaks` contains the final screened peak list.

## Interpretable evidence

| Group | Measurements | Spectral interpretation |
|---|---|---|
| G1 Morphological salience | `morph_height`, `morph_prominence` | Candidate prominence in the white top-hat response |
| G2 Local response concentration | `core_concentration`, `inner_outer_ratio` | Concentration of positive excess near the candidate centre |
| G3 Centering and symmetry | `left_right_symmetry`, `excess_centroid_offset` | Balance and centring of the local response |
| G4 Local convexity and texture | `normalized_curvature`, `edge_core_contrast`, `local_roughness_ratio` | Convexity, core contrast, and side-region texture |
| G5 Neighbour competition | `nearest_max_distance_fwhm`, `competitor_height_ratio` | Separation from and strength relative to nearby maxima |

For every scored candidate, the GUI reports the measurement value and its
signed contribution to the model margin.

## Demonstration data

`assets/examples/nai_demo_500.pkl` contains 500 example spectra with 1024
channels. `nai_demo_annotations.json` contains the corresponding fitted
reference peaks, and `sample_manifest.csv` records source IDs and count-array
hashes.

To use another data format, adapt `nai_peak/data.py` to produce the fields in
`SpectrumRecord`. The count array and detector profile are consumed by the
inference path; the remaining fields support display in the GUI.

Python pickle files may execute code when loaded. Open trusted files and verify
the checksums in `SHA256SUMS`.

## Licence and citation

The source code is available under the Apache License 2.0 in `LICENSE`. The
model weights and demonstration data are available under CC BY 4.0 in
`LICENSE-ASSETS.txt`; attribution guidance is provided in
`ASSET_ATTRIBUTION.md`. Software citation metadata are provided in `CITATION.cff`. Academic citation
requests do not add conditions to either licence.

### Software citation

When reporting research that uses this implementation, please cite the software
record below (also available through GitHub\'s "Cite this repository" menu):

> Zhang, Caolin (2026). *Interpretable local-shape screening for morphology-based
> peak detection in NaI(Tl) gamma-ray spectra* (version 1.0.0) [Software].
> https://github.com/zhangcaocao/NaI-Tl-peak-screening

The version and release date (2026-09-15) in `CITATION.cff` describe the software,
not a journal publication. Report the version actually used and, for an
unreleased or modified checkout, the commit SHA and your changes. No software
DOI is currently supplied.

### Model assets and demonstration data

For research using the bundled model weights, calibration files or demonstration
data, identify the asset paths and software version or commit used, and retain
asset attribution when sharing them. See [ASSET_ATTRIBUTION.md](ASSET_ATTRIBUTION.md)
for the covered files and a copyable attribution example. A software citation
alone does not replace the asset licence notices or modification disclosures.

### Related academic manuscript

The associated manuscript is titled *Interpretable local-shape screening for
morphology-based peak detection in NaI(Tl) gamma-ray spectra*. As of 2026-10-14,
its formal publication status and complete bibliographic details have not been
verified for this repository. No verified journal DOI or public manuscript link
is supplied here. Do not describe the software record as a published journal
article or infer manuscript authorship from the software author field. Until a
verifiable manuscript record is available, use the software citation above and
the applicable asset attribution; no separate paper citation can currently be
provided reliably.

### Updating the citation after publication

Once the publisher record is verified, maintainers should make one documentation
update that:

1. Adds a separate `type: article` entry under `references` in `CITATION.cff`,
   with the verified full author list, title, journal, publication year, DOI and
   publisher URL, plus volume, issue and pages or article number where available.
   Keep the top-level `type: software`, software authors, version and release
   date. Keep GitHub\'s default citation focused on the software; do not add
   `preferred-citation` for the article.
2. Replaces this status paragraph with the verified paper citation and DOI link,
   and requests that research using the method cite the paper in addition to the
   software version used.
3. Updates the related-manuscript note in `ASSET_ATTRIBUTION.md` to link to the
   same paper, while preserving asset attribution and licence requirements.

Validate the updated CFF against the CFF 1.2.0 schema and check all three files
for matching paper metadata. Leave unavailable fields absent rather than adding
placeholder or guessed publication information.
