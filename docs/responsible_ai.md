# Responsible AI statement

## Human review risk

Every AI output in this pipeline passes a human gate, but human review can also miss errors. We do not claim
the gates remove that risk; we limit what a mistaken approval can let through and keep every decision traceable.

- **Hard rules the reviewer cannot override.** Gate 3 cannot approve a rebuild that failed parity against the
  legacy model's outputs. Gate 4 cannot approve a validation with a red finding; amber findings and listed
  limitations need written resolution notes.
- **Errors found later can be sent back.** Review 1 (gates 1-2) reads text; some misreadings of the legacy model
  only show once the rebuild runs on data. A gate 3 or 4 reviewer can then reject and return the model to
  Review 1 with the specific issue; gate 2 reopens, and Redevelop reruns on the corrected document.
- **Limited scope.** Gate 4 approval means "validated for research" only, never production use.
- **Audit trail.** Each decision stores the reviewer, time, notes and SHA-256 snapshots of what was approved, and
  downstream stages re-verify those hashes, so any approval can be reviewed afterwards.
- **Residual risk.** A wrong approval that passes these rules is not caught automatically. Reviewer identity is
  self-declared, gates 3 and 4 may be signed by the same person, there is no revoke step after gate 4, and
  post-deployment monitoring is future work.
