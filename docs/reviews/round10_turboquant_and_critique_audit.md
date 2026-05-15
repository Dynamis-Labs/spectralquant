# Round 10 — TurboQuant comparator and critique audit

This audit accompanies the Round 10 manuscript revision. It records (i) what
"our TurboQuant implementation" actually contains, (ii) whether an external
TurboQuant archive is available in the repository or workspace for
cross-checking, and (iii) where each reviewer critique is addressed in the
paper. No new experiments are run; this round is wording and framing only.

## 1. Status of an external TurboQuant archive

Search performed in `/home/user/workspace/` and inside this repository for
files matching `turboquant*.{tar,tar.gz,tgz,zip}`, `*archive*`,
`baseline/turboquant_cutile/`, or any vendored copy of the official Google
TurboQuant code (Zandieh et al., arXiv:2504.19874).

Result: **no TurboQuant archive is present** in the workspace at the time of
this revision. The only TurboQuant-named files in the repository are
result JSONs that record the in-repository TurboQuant arm and code
references that live behind a `baseline/turboquant_cutile/` import path
that is only available on the Modal image (see
`src/spectralquant/engine.py`, lines 40–55: bootstraps a kernel-accelerated
engine when `turboquant_cutile` is importable; otherwise falls back to the
pure-Python baseline class).

Implication: the manuscript's TurboQuant comparison is against our
re-implementation of the published method, not against the official Google
reference and not against the Blackwell-cuTile port. The Round 10 manuscript
is updated to say so explicitly; previous wording such as "in-repository
TurboQuant comparator" is replaced by "our TurboQuant implementation" or
"our implementation of TurboQuant", with a code pointer to
`src/spectralquant/spectralquant.py::TurboQuantBaseline`.

## 2. What our implementation of TurboQuant contains

Our TurboQuant implementation is the `TurboQuantBaseline` class in
`src/spectralquant/spectralquant.py` (≈ lines 621–730). It is intended as a
faithful, paper-faithful realization of the data-oblivious recipe of
Zandieh et al. (arXiv:2504.19874), used as a comparator in this manuscript.
Concretely it composes:

* **Rotation.** Haar-distributed random orthogonal rotation per
  `(layer_idx, head_idx)` (`spectral_rotation.py::RandomRotation`,
  constructed via `_haar_random_orthogonal`). This matches the
  data-oblivious random rotation in the published method.
* **Scalar quantization.** A single Lloyd–Max codebook fit per
  `(layer_idx, head_idx, head_type)` using
  `LloydMaxQuantizer.fit(data_rot.flatten())` at `n_bits =
  max(1, round(avg_bits))`. Lloyd–Max fits to the empirical scalar marginal
  in the rotated basis. The TurboQuant paper specifies a per-coordinate
  scalar MSE quantizer with codebook chosen for a Beta-like / Gaussian-like
  marginal. Our implementation uses the empirical marginal directly via
  Lloyd–Max, which is at least as good as any fixed MSE codebook on the
  observed distribution and is the standard quality-control choice when the
  marginal is not known to be exactly Gaussian.
* **Johnson–Lindenstrauss residual correction.** `FullQJL` applied across
  all `d_h` coordinates (`selective_qjl.py::FullQJL`). The TurboQuant paper
  describes a 1-bit QJL transform on the residual to give an unbiased
  inner-product estimate; our `FullQJL` realizes this on every coordinate.
* **Hooked Python replay.** TurboQuant operating points in this
  manuscript's measurements run through the same Python forward-hook
  harness as SpectralQuant; both arms therefore inherit the
  `production_kernel = false` stamp, and the latency rows are not a fair
  systems comparison against the Blackwell-cuTile port that the official
  Google reference can be compiled to.

What is **not** in our implementation:
* No fused per-coordinate kernel; the round-trip is implemented in PyTorch.
* No Blackwell-cuTile / Hopper kernel. The cuTile-accelerated engine in
  `src/spectralquant/engine.py` is gated on `turboquant_cutile` being
  importable, which only happens on the Modal image, and even there it
  serves the SpectralQuant engine, not the TurboQuant baseline.
* No call into the official Google reference implementation. It is not
  vendored.

For these reasons the manuscript explicitly does not claim a strict win
over Google's reference TurboQuant or against Blackwell-cuTile; it claims a
strict win against our implementation of TurboQuant on attention-output
cosine and the four next-stage families on Qwen2.5-7B at b = 3.

## 3. Reviewer critiques and where they are addressed

Each row below names the critique, the section(s) where the critique is now
addressed in the manuscript, and the specific framing change in Round 10.

| # | Critique | Where addressed | Round 10 change |
| --- | --- | --- | --- |
| 1 | "in-repository TurboQuant" wording is awkward | Abstract, Intro, Methods under test, Results tables, Limitations, Conclusion | Replaced by "our TurboQuant implementation" / "our implementation of TurboQuant", with code pointer to `src/spectralquant/spectralquant.py::TurboQuantBaseline`. |
| 2 | Latency reality check: kernel fast (0.06 ms/tok) but hooked replay 4–40× slower than FP16 | §Latency, §Mechanism behind the latency split, §Limitations | Strengthened: explicit statement that production wins require fusing decompression into the attention tile loop; we do not claim a production speedup. |
| 3 | Limited generalizability: next-stage evidence is single-model Qwen2.5-7B single-seed | Abstract, Intro/Scope, §Limitations | Abstract bounded to Qwen2.5-7B for the four next-stage families; multi-seed and multi-architecture are explicit open work. |
| 4 | LongBench beat is brittle (5-task, qasper-driven, may regress on full 21-task) | Abstract, §LongBench Interpretation, §Limitations, §Discussion | Reframed as a directional signal on a deterministic five-task subset; full 21-task / multi-seed is open work, and we explicitly note macro could regress to or below FP16. |
| 5 | Baseline selection: comparison is to our implementation, not official Google or Blackwell-cuTile | Abstract, §Methods under test, §Limitations, §Conclusion | Wording moved from "in-repository TurboQuant" to "our implementation of TurboQuant"; explicit statement that production-default claims need the official optimized baseline. |
| 6 | Gaussian rate–distortion derivation: keys are not shown Gaussian | §Background (water-filling intuition), §Method (water-filling rule), §Limitations | Clarified that the Gaussian rate–distortion model is a design heuristic from rate–distortion theory; Lloyd–Max handles the empirical scalar marginal MSE-optimally without requiring Gaussianity of the source. |
| 7 | "Greedy matches LP optimum" / "within one bit per dimension" needs proof or softening | §Method (greedy integer projection) | Softened: dropped the unproven "matches the LP optimum" claim. We now describe the greedy projection as the implementation choice that is consistent with the matroid structure of the relaxed problem; a precise approximation guarantee is not asserted in this manuscript. |
| 8 | Selective QJL needs ablations (eigenbasis+uniform, eigenbasis+waterfill, eigenbasis+waterfill no QJL, random rotation+same Lloyd–Max, raw basis+waterfill) | §Limitations, §Discussion | Added an explicit "ablation gap" item: the selective-QJL contribution is not isolated by the present evidence; only the rotation contribution and the allocation contribution are isolated by the three-way matrix. |
| 9 | Value-side of KV compression under-discussed | §Method (pipeline overview), §Limitations | Clarified that V is rotated by the same per-(layer, head) eigenbasis as K and quantized with the same per-dimension Lloyd–Max codebooks; we explicitly flag the unresolved question whether V benefits from a separate calibration object as open work. |
| 10 | Model coverage uneven (spectral observation broad; downstream evidence narrow) | Abstract, Intro, §Limitations | Abstract names the spectral observation as cross-model and the next-stage families as Qwen2.5-7B-only; intro reaffirms the bound. |

## 4. Recentering on four core claims

Round 10 recenters the abstract, introduction, and conclusion on four core
claims:

1. **Empirical law.** KV-key covariance is low-effective-rank across the
   tested LLMs (Qwen2.5 family, Mistral-7B-v0.3, Llama-3-8B): participation
   ratio in [4, 6] at d_h = 128.
2. **Algorithm.** Calibrated eigenbasis rotation plus water-filled bit
   allocation inside the semantic subspace, with per-dimension Lloyd–Max
   codebooks, hooked into a Python replay harness that is quality-valid
   before it is systems-valid.
3. **Mechanistic evidence.** Attention-output cosine improves at matched
   compression: + 0.27 to + 0.38 over our implementation of TurboQuant on
   the four three-way operating points, and + 0.018 over the
   uniform-allocation special case at b = 2 on Mistral.
4. **Downstream sanity checks.** Perplexity, greedy-generation token-F1,
   and a deterministic five-task LongBench subset on Qwen2.5-7B at b = 3
   remain plausible; production-kernel latency, multi-seed bands, full
   21-task LongBench, multi-architecture coverage, and an official Google
   / Blackwell-cuTile baseline are all named as future work.

## 5. Items that this audit explicitly does not unblock

* Multi-seed bands.
* Production-kernel end-to-end latency.
* Official Google TurboQuant or Blackwell-cuTile head-to-head.
* Full 21-task LongBench.
* Multi-architecture next-stage families.
* Selective-QJL ablation matrix.
* Value-side calibration audit.

These remain explicit open work in §Limitations and §Discussion / Systems
roadmap.
