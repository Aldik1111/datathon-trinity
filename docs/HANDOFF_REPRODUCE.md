# Local handoff preparation and reproducibility

This is a local reproduction recipe for the experimental model and handoff integrity checks. It processes only the extracted packet and MOV directory explicitly supplied by the operator. Do not upload packet contents, videos, extracted frames, manifests, caches, or reports to cloud services. The commands below do not publish, deploy, or promote a model.

## Source snapshot and environment

The historical model baseline is upstream `feature/local-safe-recognition` at commit `0f15815b49d3b0e7349aa8d5bd13679b166102b9`. The current local working branch is `feature/handoff-diagnostics-20261018`, based on that commit; the new preparation/reproduction commands are part of this branch's changes and are not present in commit `0f15815` itself. Reproduce from the current diagnostics branch with those changes applied. The historical 185-test result came from a separate pristine checkout; it does not include the new preparation scripts. To recreate that comparison checkout from the repository root, use `git worktree add ..\trinity-baseline-0f15815 0f15815b49d3b0e7349aa8d5bd13679b166102b9`, then run its own pinned environment and test suite there. Keep subsequent experiments in new artifact directories; do not overwrite the extracted baseline or original scenario model.

Use Python 3.12 and the repository's pinned requirements. The pinned packages are `opencv-python==4.14.0.94`, `fastapi==0.143.3`, `uvicorn==0.54.0`, `python-multipart==0.0.32`, `streamlit==1.66.0`, `requests==2.34.2`, `python-dotenv==1.2.4`, `pillow==12.3.0`, `numpy==2.5.4`, `scikit-learn==1.9.1`, `av==19.0.1`, `httpx==0.28.1`, and `pytest==9.1.1`. The ML and dev requirement files include the base requirements file. Install all packages into the same local environment:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt -r requirements-dev.txt
```

Set three task-specific paths to the already-extracted packet, the authorized local MOV folder, and a new output folder under this repository's ignored `artifacts/` directory:

```powershell
$Repo = (Get-Location).Path
$Packet = 'C:\path\to\extracted\Trinity-handoff'
$VideoDir = 'C:\path\to\authorized\videos'
$Out = Join-Path $Repo 'artifacts\handoff-run-20261018'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
```

The packet and video paths are supplied at run time; no personal path is embedded in the script. The preparation command verifies every entry in the packet's `manifest-sha256.csv`, matches only local MOVs by SHA-256, and extracts only `datathon-trinity/artifacts/` members from the baseline archive. It refuses unsafe archive paths, a changed baseline, an output inside either input directory, output not ignored by Git, or output provenance that differs from a prior run. A changed packet, annotation manifest, or local path requires a new `$Out` directory. Standard output contains only aggregate counts and checksums; private clip or path details are not printed.

```powershell
& $Python scripts/prepare_handoff.py --packet $Packet --video-dir $VideoDir --output $Out
```

The generated manifest contains local paths and annotation metadata and must remain inside ignored `$Out\handoff-input\`. Do not commit it. The baseline model is preserved at `$Out\handoff-baseline\scenario-model\` and the baseline safe model at `$Out\handoff-baseline\safe-model\`. `integrity.json` records aggregate counts and input/model digests for reproducibility. Those hashes are technical integrity evidence; they do not establish legal rights, rule applicability, or compliance.

## Rebuild and compare in new directories

Use new cache and model names for each experiment. The initial frame cache uses the frozen backbone and the original temporal feature version. The residual cache is a deterministic per-clip transform of those cached features and does not reread the MOVs or use labels to fit parameters.

```powershell
$Input = Join-Path $Out 'handoff-input'
$Baseline = Join-Path $Out 'handoff-baseline'
$Features = Join-Path $Out 'scenario-features-repro-v1'
$ResidualFeatures = Join-Path $Out 'scenario-features-residual-repro-v1'
$ResidualModel = Join-Path $Out 'scenario-model-residual-repro-v1'
$BaselineModel = Join-Path $Baseline 'scenario-model'

& $Python scripts/extract_scenario_features.py `
  --manifest (Join-Path $Input 'manifest.json') `
  --backbone (Join-Path $BaselineModel 'backbone.onnx') `
  --output $Features

& $Python scripts/derive_residual_cache.py `
  --source $Features `
  --output $ResidualFeatures

& $Python -m src.training.train_scenarios `
  --manifest (Join-Path $Input 'manifest.json') `
  --feature-cache $ResidualFeatures `
  --output $ResidualModel

Copy-Item (Join-Path $BaselineModel 'backbone.onnx') $ResidualModel
Copy-Item (Join-Path $BaselineModel 'BACKBONE-LICENSE.txt') $ResidualModel

& $Python scripts/diagnose_scenarios.py `
  --manifest (Join-Path $Input 'manifest.json') `
  --cache $Features `
  --model "baseline=$BaselineModel" `
  --model "residual_repro_v1=$ResidualModel" `
  --output (Join-Path $Out 'diagnostics-residual-repro-v1')
```

Diagnosis is a local before/after analysis on the known development corpus, not a fresh independent evaluation and not model-selection evidence for deployment. Keep its CSV/JSON outputs in the ignored output tree. Preserve the baseline model, comparison, source annotations and original media unchanged.

## Local HTTP and test verification

The HTTP smoke creates only synthetic geometric MOVs and a temporary local API database/upload directory under a new output path. It uploads synthetic MOV bytes over local loopback HTTP and checks the selected local model, auth, mode routing, cleanup and deterministic response to a renamed synthetic input. No source video is uploaded and no cloud service is used.

```powershell
$HttpOutput = Join-Path $Out 'http-smoke-residual-repro-v1'
& $Python scripts/verify_handoff_http.py --model $ResidualModel --output $HttpOutput
& $Python -m pytest tests -q --basetemp .pytest-reproduce-tmp
```

To exercise the real HTTP queue on the authorized source clips, use the separately reviewed local batch workflow only if that work is in scope and its report remains in ignored artifacts. This document's synthetic HTTP smoke is not a substitute for an independent quality evaluation. Neither smoke nor test success authorizes production use, deployment, or promotion of a candidate model.
