# Archive

Historical code retained as a relic, kept off the Python path so it
doesn't interfere with active development. Restore by `git mv` back
to the working tree if ever needed.

## archive/v7/

The v7 generation of YAML-BERT, superseded by v8 in May 2026. v7's
architecture used hybrid bigram/trigram compound prediction targets
(simple_head + kind_head, ~28K vocab). v8 replaced this with atomic
prediction conditioned on doc_vec from a bottom-up tree aggregator.

See `docs/v8-276K-scaleup-results.md` for the v7→v8 comparison.

## archive/v8-phase0/

Earlier v8 training scripts (train_v8_phase0.py, eval_v8_phase0.py)
that were superseded by scripts/train.py and scripts/eval_probes.py.
Functional but not the canonical pipeline.
