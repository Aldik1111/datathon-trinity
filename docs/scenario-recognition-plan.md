# Scenario recognition implementation plan

> For agentic workers: use focused subagent implementation and independent review.

**Goal:** Train experimental candidates for all table types, compare all videos honestly, and provide manual posttraining inference.

**Architecture:** Frozen local MobileNetV2 features feed temporal logistic heads; grouped evaluation is offline, while the same feature/head code serves uploads. Additive API mode and UI preserve safe recognition.

**Spec:** docs/scenario-recognition-design.md

## Global constraints

- Original annotations/media remain read-only. No cloud processing or label lookup during inference.
- Weak/unclear causes remain explicit; unsupported folds produce null scores.
- Fixed C=0.01, threshold=0.65; no evaluation tuning. Byte-identical groups never cross a training/evaluation boundary.
- Scenario model has numeric NPZ head, checksum-validated ONNX backbone and JSON metadata; no executable pickle.

## Review focus

- Singleton class held out: unsupported, not a true negative.
- Duplicate rows with different types: union fitting labels, preserve conflict in report.
- Uploaded renamed original: identical pixels produce identical predictions.
- Switching selected input/mode: old result remains clearly attributed to its run.
- Request failure or timeout: media, ROI and analysis sidecars are cleaned.

## Tasks

1. Data/training: add scenario_dataset.py and train_scenarios.py; tests for partial labels, duplicate grouping, fold isolation, unsupported folds and aggregate metrics. Inputs canonical weak manifest and feature cache; outputs final numeric head, metadata/report, held-out and final comparisons. Run focused tests, then all tests at integration.
2. Root features/inference: add scenario_features.py and scenario_classifier.py with tests first. Extract features with versioned preprocessing; use real PTS; score windows; return all types and candidate evidence. Shared arrays and metadata interface follows the spec. Verify model checksum and renamed-file invariance.
3. API: add allowlisted analysis_type, scenario_model_path setting, analysis sidecar dispatch/cleanup, scoped result validation/public response. Test bad modes, malformed metadata, default compatibility and cleanup before implementation.
4. UI: add scenario mode with local 42-video chooser/upload, progress, model support, all class scores and evidence, result provenance and offline comparison view/download. Existing safe mode remains available. Test with AppTest and HTTP client tests.
5. Integration: extract all originals once with content+feature-version cache; train/evaluate/final-fit; build comparison files. Inspect actual scores and errors without tuning. Run whole-video HTTP and browser, full suite, independent review. Package code+model and update instructions/launchers.
