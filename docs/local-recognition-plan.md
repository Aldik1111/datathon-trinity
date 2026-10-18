# Local safe-state recognition — 2026-10-18

User approved preparing recognition on this PC because the previous prototype is unavailable. Reuse the current main service and preserve all original annotation/video data. Work on feature/local-safe-recognition in a fresh checkout.

## Scope and design

First usable model: classify an explicitly selected safe region as open or closed. Low confidence becomes unknown; there are no labeled unknown examples, so this is abstention, not a trained third class. Use CPU HOG/color features and regularized logistic regression for a fast reproducible baseline. This is genuine learning from the provided examples, not label lookup. It does not detect people, identity checks, transfers, locks, or regulatory violations.

Different videos have different camera positions. The user selects a safe region, with presets and a preview. Known example videos may have a preconfigured static ROI, which is calibration only; inference must never read event/state labels or use a video identifier to choose predictions.

Export only independently labeled safe frames with clear visibility; exclude interpolated frames and ambiguous state labels. Source video groups and byte-identical copies stay in one split. Fixed deterministic split before training; train selects weights, validation selects regularization/abstention, test is evaluated once without tuning. Report per-class precision/recall/F1, confusion matrix, sample/video counts, coverage, and same-camera limitations. Sparse frame labels do not establish event interval ground truth.

Preserve existing demo/API behavior. New analyzer output uses additive verdicts review_required and no_event with analysis_scope=safe_state. no_event means no confident open state in sampled frames, never absence of all violations. Open/uncertain events include actual PTS, start/end, and JPEGs. UI displays honest scope, timeline, metrics, and downloadable result.

## Shared interfaces

Dataset manifest JSON: {schema_version:1, samples:[{image_path:absolute str,label:open|closed,split:train|val|test,video_id:str,filename:str,group_id:sha256,timestamp_sec:float,roi:[x,y,w,h]}], videos:[{video_id,filename,path:absolute str,group_id,split,roi:[x,y,w,h]}], exclusions:[...], source_sha256:str}. Images are RGB crops saved as PNG. Extraction respects MOV display orientation.

Model module src/vision/safe_model.py: features(image_bgr: ndarray)->ndarray; load_model(model_dir)->object; predict_crop(model,image_bgr)->dict with state(open|closed|unknown), confidence(float 0..1), open_probability(float 0..1). Numeric NPZ + JSON artifacts; no pickle. Training CLI: python -m src.training.train_safe --manifest PATH --output DIR. Saves model.npz, metadata.json, report.json. Metadata has threshold and model_version. Dataset is read-only.

Video inference src/vision/safe_classifier.py: analyze_video(video_path:str)->dict. Reads model from MODEL_PATH and required ROI from video_path+'.roi.json' ({roi:[x,y,w,h]}). No ROI => clear failure. Shared video helper src/vision/video_io.py: iter_sampled_frames(path,sample_fps=2) yields (timestamp_sec, BGR image), applying orientation and real relative PTS. max 300s/1200 samples. Raw result includes verdict, events, analysis_scope='safe_state', model_version, summary, observations [{timecode_sec,state,confidence}]. Raw events use existing event_type/timecode_sec/description/evidence_frames/jpeg_bytes plus start_sec/end_sec/confidence. At most 100 events, 3 JPEGs/event. Closed-only observations => no_event; open or uncertain => review_required.

API POST /api/v1/analyze accepts optional multipart roi JSON list, validates finite normalized positive bounds before queueing, saves private per-job ROI sidecar. Cleanup all termination paths removes video and sidecar. Existing no-ROI demo clients continue working. Adapter validates the new scoped output. Public result preserves validated fields.

## Tasks and verification

1. Data export: implement tests for duplicate grouping, label filtering, frame orientation/time and invalid boxes. Export existing snapshot and original MOVs read-only. Generate contact sheets and inspect them.
2. Model/inference: test train/eval split isolation, saved-model reproducibility, invalid inputs, uncertainty and temporal grouping. Train baseline and report untouched test performance, regardless of quality.
3. API/UI integration: test scoped verdict semantics, ROI validation and cleanup, display of open/closed/uncertain results. Add ready examples using manifest ROI and show each split. Add metrics and result download.
4. Root verification: baseline and full pytest, independent code review, one held-out whole MOV through real HTTP queue, browser interaction and screenshot. Provide Windows start/stop launchers, local URL, report, code patch and concise usage instructions. No cloud processing or public deployment.

## Progress

- Fresh main cloned; branch created; dependencies installing.
- Ruling: use an explicitly calibrated ROI because camera positions differ. No automatic object localization claim.
- Ruling: proceed with the user-authorized preparation and existing first-safe scope; avoid another permission round for implementation details.
