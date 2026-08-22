# Release checklist

- [ ] Replace repository-owner metadata if a remote URL is added.
- [ ] Run `python tools/audit_release.py`.
- [ ] Run `pytest` and `ruff check .`.
- [ ] Build with `python -m build` and inspect both artefacts.
- [ ] Confirm that no dataset, checkpoint, patient/specimen key or acquisition
      path is tracked.
- [ ] Confirm that no nested Git repository or generated report is tracked.
- [ ] Verify third-party dataset and checkpoint licences separately.
- [ ] Tag the exact version referenced by a report or preprint.
- [ ] Publish aggregate results only after rechecking their frozen protocol.
