# HMDA visual redesign concepts

Generated with the built-in image-generation tool, 2026-11-06. Full generation
and refinement prompts are in prompts.json. These are visual concepts, not
implemented screens or verified new functionality. No app code or model artifacts
were changed for this exploration.

1. **Research workbench** — compact cohort controls, small-multiple outcome charts,
   continuous table layout, quiet navy accents. Best for everyday data exploration.
   Annual chart lines are illustrative, not verified annual metrics. The first
   draft's action-label error was corrected: origination is action 1 only.
2. **Monitoring report** — editorial hierarchy, document outline, metric table,
   confusion matrix and review notes. Best for formal review and reporting.
3. **Investigation workspace** — evidence and analysis side by side, contextual
   agent activity, explicit citations and an evidence index. Best overall direction
   for the analyst copilot; combine with concept 1's cohort explorer.

Model values shown are from the existing saved-model evaluation in
outputs/reports/gradient_boosting_evaluation_2025.json, not the new training run.
Agent conversation in concept 3 is explicitly an example. Saved investigations,
export controls, evidence collection and saved views are proposed interactions,
not claims that those features are already implemented. The threshold would be
read-only for saved holdout evaluation, even where a mockup control looks editable.

Across all directions: replace repeated rounded cards with ruled sections, remove
marketing hero banners, put definitions and provenance beside evidence, use compact
metrics instead of oversized KPI tiles, and keep tables/charts central.
