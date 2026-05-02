# Third-Party Notices

This repository incorporates source code from the following projects. The
LatentTSF-specific code (top-level `my_*.py`, `run_*.sh`, the latent-space
training logic, the pretrained AE checkpoints, and this `README.md`) is
released under the MIT License (see `LICENSE`). The third-party portions
listed below retain the licenses of their original projects.

---

## Time-Series-Library (TSLib)

- **Upstream:** https://github.com/thuml/Time-Series-Library
- **License:** MIT (see upstream repository for the canonical text)
- **Files derived from or adopted from TSLib (used as-is or with light edits):**
  - Most of `models/` (the baseline forecasting models such as `Autoformer.py`,
    `iTransformer.py`, `PatchTST.py`, `DLinear.py`, `TimesNet.py`, etc.)
  - Most of `layers/`
  - Most of `data_provider/` (excluding `m4.py`, see below)
  - Most of `exp/`
  - Most of `utils/` (excluding `losses.py` and `m4_summary.py`, see below)
  - `run.py`

The LatentTSF training pipeline (`my_train.py`, `my_utils.py`, `my_AE.py`,
`my_MAE.py`, `my_temporal_AE.py`, `RevIN.py`, `run_train.sh`, `run_ae.sh`)
was authored for this project and is MIT-licensed.

## N-BEATS (Element AI Inc.)

- **Upstream:** https://github.com/ElementAI/N-BEATS
- **Paper:** Oreshkin et al., *N-BEATS: Neural basis expansion analysis for
  interpretable time series forecasting*, https://arxiv.org/abs/1905.10437
- **License:** **Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0)**, https://creativecommons.org/licenses/by-nc/4.0/
- **Copyright:** © 2020 Element AI Inc.
- **Files retained verbatim with the original Element AI license header
  (NOT covered by the MIT license of this repository):**
  - `utils/losses.py`
  - `utils/m4_summary.py`
  - `data_provider/m4.py`

> **Important:** the CC BY-NC 4.0 terms forbid commercial use of those three
> files. They were inherited via TSLib and are used here for the M4
> short-term-forecasting benchmark / N-BEATS-style loss functions.
> If you intend to use this codebase commercially, you must replace or remove
> these three files, or obtain an explicit commercial license from
> Element AI Inc.

---

If you reuse this code, please cite the LatentTSF paper as well as TSLib and
N-BEATS where appropriate.
