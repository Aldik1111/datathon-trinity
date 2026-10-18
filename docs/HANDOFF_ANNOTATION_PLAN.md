# Annotation plan for action and track training

## Readiness

The current annotation structure can support a first action and track experiment: event records carry a type, start and end times, an assessment, coverage, a session identifier, and object track references. Box records carry track identifiers, object classes, and frame times. Before using these labels as training truth, review the semantics below and add independent adjudication for ambiguous or consequential events. A populated field does not by itself make an annotation confirmed.

Action labels should be counted by independent source video group, not by event count. Repeated events in one video and duplicate copies are not independent examples. Split by source video group before fitting or selecting any model settings. The already discussed diagnostic clips belong to the known development corpus. They may participate in exploratory grouped evaluation with their whole group excluded from each fold's fit, but must never be described as an untouched independent test or used as inference inputs.

## Label each event as an observation

For every action, record a stable `type` identifier, `start_sec`, `end_sec`, `session_id`, `coverage`, and `object_ids` that point to one or more `track_id` values. Use the shortest interval that captures the visible action, and keep repeated actions as separate events when their boundaries are clear. Link only tracks that participate in the action. Track identifiers should keep the same physical object or person identity through the session, and a track should not change its object class midway without an explicit correction.

Use `assessment` to distinguish evidence strength:

- `observed`: a recorded assertion that the action or state is visible. Use as an action-positive target only after human review/provenance is established; a stored value alone does not establish review.
- `normal`: the annotator considered the episode or behavior normal. This is **not** absence of the named action. For example, normal cash counting is still a positive example of counting. Never turn this field directly into action-negative supervision.
- `candidate`: a possible event needs a second review. Keep it out of confirmed positive and negative targets until adjudicated.
- `unclear`: the view or evidence cannot settle the label. Treat it as unknown.

Use `coverage=full_interval` only when the entire stated interval was reviewed and the relevant action could be seen. Use `coverage=partial` when a camera cut, occlusion, obstruction, missing context, or limited review leaves part of the interval unresolved. Partial coverage can support a positive event when the action itself is directly visible, but it cannot support a negative claim about the unseen portion.

For a new action-training export, add a separate adjudicated `action_presence` (`present`, `absent`, `unknown`) for each target action and interval, plus reviewer/version fields. These are proposed export fields, not fields currently implemented in the labeling website. `absent` requires an explicitly reviewed, sufficiently visible interval for that specific action. Normal behavior and absence of an action answer different questions. Preserve the original JSON and assessment fields when preparing this export.

## Separate “not visible” from “did not happen”

Record a visible observation separately from completeness of the episode. For example, “the document is not visible in these frames” is an observation; “no identity check occurred during the full transaction” requires review of the complete relevant episode and adequate camera coverage. When the episode is incomplete, mark the action unknown rather than normal. Absence from a sparse sample, a single frame, or a camera view does not establish absence from the real-world event.

Store legal qualification in a separate reviewer field, such as `rule_assessment`, with values like `unreviewed`, `not_assessed`, or an adjudicated disposition tied to a dated procedure and its required facts. Visual labels should describe what the recording shows. They should not encode a legal violation, employee intent, authorization, transaction status, or compliance conclusion unless a qualified reviewer has separately established the required non-video facts and approved the rule mapping.

## Resolve and verify labels before model use

For each class, track counts of confirmed positive video groups, confirmed full-episode negative groups, unclear/candidate groups, and groups with partial coverage. Keep these counts separate from event and frame counts. A class with only one independent positive group is not ready for generalization claims; report it as unsupported or use it only for a clearly marked training diagnostic.

Adjudicate disagreements before fitting. Record the reviewer, annotation version, and reason for each correction. Check interval bounds against video duration, session continuity, object references against existing tracks, and track class consistency within each video. Keep a change log so corrected labels do not silently alter prior evaluations.

Any sequence model should consume only image/track/action evidence available at inference time. Filenames, source descriptions, table row numbers, user notes, and timecodes that point to known labels must not be model features. Select transforms and thresholds on training/validation groups only, then evaluate once on independent held-out groups. Report per-class support by independent video group, coverage, abstentions, and the exact positive/negative label policy.
