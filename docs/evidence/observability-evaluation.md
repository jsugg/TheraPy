# Observability evaluation protocol (O3.4)

These evaluators are regression tools over versioned synthetic evidence. No
telemetry, deterministic verdict, owner score, or future LLM-judge score is
evidence of therapeutic benefit.

## Evaluator dimensions

| Plan dimension | Evaluation path |
| --- | --- |
| Language adherence | Deterministic Spanish/English stopword check for `language-01`; judge or owner review for broader adherence |
| Persona/policy adherence | LLM judge with reported uncertainty plus owner review |
| Crisis safety | Deterministic non-empty/configured-resource-marker gate; owner review for context, tone, and escalation quality |
| Non-diagnostic and diagnosis/medication boundaries | Deterministic case-insensitive deny terms; owner review for the complete boundary behavior |
| Helpfulness/tone | LLM judge with reported uncertainty plus owner review |
| Hallucination | Evidence-linked judge or owner review; never deterministic auto-pass |
| Retrieval grounding/citation | Retrieval assertions plus judge or owner review against the retrieved passage |
| Memory attribution/fidelity | Judge or owner review against the exact memory evidence |
| Tool authorization/correctness | Deterministic authorization records where available plus owner review |
| Longitudinal claim support | Owner review against exact cited history |
| Structured-output validity | Deterministic JSON parsing for `structured*` cases; declared-schema checks can be added when a schema is present |
| Response completeness | LLM judge with reported uncertainty plus owner review |

Every prompt, model, or policy change runs the golden set. A failed
deterministic check on a high-risk case sets `human_review_required`; dimensions
without a deterministic pass rule remain
`requires_llm_judge_or_owner_review`.

## Speech protocol

`evaluate_speech.py` consumes transcripts produced by a separate STT run; it
does not invoke Whisper. JiWER 4 computes per-case and corpus WER, CER, MER, and
WIL after explicit normalization: `str.lower()`, replace every Unicode
punctuation character with a space, collapse whitespace, and preserve letters
and accents. Empty references are silence-detection cases: any non-empty
hypothesis is a false-speech detection, and silence is excluded from all word
and character error rates.

WER/CER claims are permitted only when every scored fixture has owner review
status `approved`. The CLI otherwise exits 2. `--allow-seeded` exists only for
exploration: its report is labeled `seeded-not-reviewed` and states that WER/CER
claims are not valid. Unreviewed runtime transcripts must never be scored.

## Owner TTS listening

Generate the same configured voice/speed output for each committed
`tts_listening_phrases` entry:

1. `tts-en-01`: “Let's take a moment to notice what went well today.”
2. `tts-es-01`: “Vamos a repasar juntos cómo te sentiste esta semana.”
3. `tts-pt-01`: “Vamos revisar juntos como você se sentiu esta semana.”

The owner listens under the same device, volume, and environment and records,
per phrase, the fixture hash, build/model/voice/speed, date, **pronunciation
1–5**, **naturalness 1–5**, and a short note. Repeat after TTS/model/policy
changes. These are one owner's repeatable listening scores and must never be
labeled population MOS.

## On-demand profiling

There is no continuous profiler. Use `py-spy dump` for an external CPU snapshot
and Memray on demand for Python/native allocation diagnosis. Enable ONNX Runtime
JSON operator/thread/latency profiling for Kokoro or FastEmbed only in an
isolated, sanitized benchmark after broad traces localize a regression. Compare
profiler-off/on, restrict and promptly delete artifacts, and never export
profiling continuously.

## Commands

Prepare a complete `{case_id: transcript}` mapping, then run reviewed speech
evaluation (currently seeded fixtures require owner approval first):

```bash
.venv/bin/python scripts/observability/evaluate_speech.py \
  --hypotheses .local/obs-eval/speech-hypotheses.json \
  --output .local/obs-eval/speech-report.json
```

For an explicitly invalid-for-claims seeded dry run, append `--allow-seeded`.

Prepare a complete `{case_id: assistant_response}` mapping and run behavior
checks:

```bash
.venv/bin/python scripts/observability/evaluate_behavior.py \
  --responses .local/obs-eval/behavior-responses.json \
  --output .local/obs-eval/behavior-report.json
```

With the local `llm-observability` Phoenix service running, upsert datasets from
the pinned spike environment:

```bash
.local/obs-spike/phoenix-venv/bin/python \
  scripts/observability/phoenix_datasets.py \
  --endpoint http://localhost:6006
```
