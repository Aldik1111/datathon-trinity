# Experimental table-scenario recognition

The user requests recognition of the remaining table scenarios, training, all-video comparison against their annotations, and manual inference after training. Preserve original media, annotations and the existing safe-state mode. This is a new subsystem within the existing local app.

The available table contains 42 videos (40 byte-distinct groups), 22 positive rows, 20 negative rows, and 18 proposed scenario types. Several causes are disputed and most types have only one positive media group. The product must therefore present learned scenario candidates, not verified procedural violations. Identity verification, report contents, network disconnection, cash ownership and lock operation cannot be established by appearance alone.

## Method

Use the official ONNX MobileNetV2 image backbone locally through OpenCV DNN. Freeze it; extract its 1280-dimensional global average pooling features from aspect-preserved RGB frames at 1 FPS, normalized with the documented ImageNet mean/std. Project features with a fixed seed-independent-of-data matrix to 256 dimensions. Construct temporal descriptors from the current frame, mean in a +/-2-second context, and future-minus-past context (768 dimensions). Do not use filename, media identity, absolute/normalized timestamp, annotation geometry, or table label as a predictor.

Train separate regularized logistic heads using weak positive windows within +/-3 seconds of supplied table anchors. Normal-table videos provide negatives; missing types in other positive videos are unknown, not automatic negatives. Duplicate rows share a single media group; union their positive anchors for fitting and retain their disagreement in comparison. Fit scaling on each training partition only. Fix C=0.01 and decision threshold=0.65 before evaluation; do not tune against evaluation results.

Evaluate leave-one-media-group-out. Classes absent from a training fold are unsupported, with null scores, never negative predictions. Compare each canonical row to its held-out prediction; provide type precision/recall and coarse anchor tolerance +/-5 seconds, denominators, abstentions, conflicts and source review status. Sparse table anchors are not interval ground truth. Train a separate final head on all available weak labels for deployment. Label results on that training set as resubstitution and keep them separate from held-out results.

## Inference and UI

The manual user flow selects scenario mode, chooses any of all 42 local originals or uploads a MOV, then presses Analyze. No prefilled answers or table-assisted inference. Preserve safe-state mode and ROI validation. Scenario results list all types, model scores, support, candidate/not_flagged/unsupported decisions, and representative frames at peak scoring windows. Time intervals indicate scored windows, not proven onset/end of a banking action. No-candidate means no score passed the threshold, not no violations.

API multipart analysis_type is an optional allowlisted safe_state/bank_scenarios field. Omission preserves existing behavior. A per-job private analysis sidecar persists the selection; cleanup removes it on all paths. Scenario model path comes from server SCENARIO_MODEL_PATH; clients cannot provide paths or callables. The existing MODEL_PATH remains the safe model.

Raw bank_scenarios result: verdict review_required/no_candidate, events (same evidence structure as safe, with start_sec/end_sec/confidence), model_version, feature_version, score_semantics='uncalibrated_model_score', summary {sample_count,sample_fps,events_truncated}, class_results [{type,title,score,threshold,decision,positive_groups,negative_groups,peak_sec}], limitations list[str]. Unsupported score/peak are null. Candidate decisions must match event types; no_candidate iff no candidates/events. Each event has 1-3 JPEGs. Normalized output uses violations instead of events as before. Metadata validation bounds strings/lists, rejects NaN and unknown types/decisions, validates scores/times/counts and consistency.

## Delivery and verification

Test first: group isolation/unknown labels/unsupported classes; identical feature transforms in training and inference; renamed input invariance; adapter validation/dispatch/cleanup; UI mode/42-file selection/provenance/comparison. Run all 42 through frozen feature extraction and held-out/final models, and a real uploaded movie through the HTTP queue and browser. Save a standalone comparison JSON/CSV/HTML, all individual final results, model report, portable source+weights, updated launchers and Russian instructions. Run full tests and independent review before claiming completion.

Authorization ruling: the user's direct implementation request and developer autonomy instructions authorize these reversible local changes; proceed without additional skill-stage permission gates. No cloud video processing, public deployment, or source annotation edits.
