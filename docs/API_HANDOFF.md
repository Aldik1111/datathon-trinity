# API handoff and contract readiness

The local API accepts one MOV upload at `POST /api/v1/analyze` as multipart field
`file`, with `X-API-Key`. It returns HTTP 202 and a `job_id`. Poll
`GET /api/v1/jobs/{job_id}` with the same header; `status` is `processing`,
`done`, or `failed`. Queue waiting counts toward the configured analysis deadline,
which is limited to 300 seconds. A failed analysis stays failed; the API does not
turn decoding or model errors into a negative verdict.

`GET /api/v1/readiness` is authenticated and reports the configured default route,
whether omitting multipart `analysis_type` is routed to `jury`, `safe_state`, or
`bank_scenarios`, whether model weights are configured, and whether the default
is configured for the jury contract. `jury_contract_compatible` is false for
known experimental/demo configurations and null for an unverified custom legacy
analyzer. This endpoint does not attest previous jobs: only each job's output is
checked against its selected contract. `model_load_verified` is always
false because this endpoint does not load a model. Contract compatibility does
not certify model quality.

When `DEFAULT_ANALYSIS_TYPE` is unset, the API infers the omitted-field route
from a known configured safe-state or scenario analyzer; otherwise it keeps the
old configured-analyzer jury route. Set it to `jury`, `safe_state`, or
`bank_scenarios` to choose a route explicitly. Explicit `jury` rejects a known
experimental classifier with HTTP 409. A request may select either experimental
mode by sending one multipart `analysis_type` field. The browser UI sends its
selected experimental mode explicitly, including safe-state uploads that
predate scenario selection. Safe-state requests require the ROI field. Scenario
mode selected explicitly uses its separate scenario model configuration; an
omitted mode inferred from a legacy configured scenario analyzer keeps that
analyzer's existing `MODEL_PATH` routing.

The jury contract remains `violation` or `no_violation`. Safe-state uses
`review_required` and `no_event`; scenario analysis uses `review_required` and
`no_candidate`. These verdicts express different evidence scopes. They are
returned with their `analysis_scope` and are never rewritten into confirmed
binary judgments. If a known experimental analyzer is configured as the jury
default, the API rejects omitted-mode uploads with HTTP 409. Select an explicit
experimental mode or configure a jury-compatible analyzer.

The scenario scores are uncalibrated model scores and scenario candidates require
human review. `no_candidate` means no candidate crossed the configured threshold
in the supported classes; it does not establish that no violation occurred.
Similarly, `no_event` means no open or uncertain safe state was observed in the
sampled frames, not that all possible violations are absent.
