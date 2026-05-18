# Reproducibility Checklist

## Before running

- [ ] Code repository commit recorded.
- [ ] Environment recorded.
- [ ] Dataset files hashed.
- [ ] Config file written.
- [ ] Random seed fixed.
- [ ] Output directory is clean.

## During running

- [ ] Command logged.
- [ ] GPU type logged.
- [ ] Wall-clock time logged.
- [ ] Training loss logged.
- [ ] Evaluation logs saved.

## After running

- [ ] Raw predictions saved.
- [ ] Metrics JSON saved.
- [ ] Config copied into results directory.
- [ ] Run manifest completed.
- [ ] Dataset hashes included in manifest.
- [ ] Any failure documented.

## Before reporting

- [ ] Tables regenerated from metrics files.
- [ ] No manual table edits.
- [ ] Random subset IDs persisted.
- [ ] Number of seeds/draws shown in table.
- [ ] Mean/std/CI shown where applicable.
- [ ] Claims updated to match evidence.
