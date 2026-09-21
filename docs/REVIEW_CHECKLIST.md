# Experiment review checklist

- [ ] Install the locked environment and run lint, formatting, and tests.
- [ ] Rebuild from bundled raw files in a clean workspace; inspect any processed-data differences.
- [ ] Confirm target definition, units, source coverage, and source provenance caveats.
- [ ] Check every feature against its availability at a monthly forecast origin.
- [ ] Keep development and holdout forecast dates disjoint; retain simple benchmarks.
- [ ] Regenerate results with `korea-tourism reproduce`; review metrics, predictions, selection, plots, and manifest together.
- [ ] Verify written conclusions agree with generated tables, including negative findings.
- [ ] Check Markdown links and ensure README images are included in the change.
- [ ] Inspect `git diff --check` and the final file list for secrets, local caches, or unrelated artifacts.
- [ ] If the holdout influenced new choices, disclose that and plan a fresh evaluation.

This checklist documents review expectations. It does not initialize repositories, change remotes, or publish automatically.
