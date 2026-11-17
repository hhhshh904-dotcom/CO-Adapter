# Reproducibility and review checklist

## Scope

This is a compact CO-Adapter code release, not a dump of the experimental server.
It intentionally excludes process supervisors, GPU allocation histories, restart
copies, private paths, raw data, pretrained weights, checkpoints and unfinished
manuscripts. Mixed queries are an optional controlled variant, not a second
research framework. Old cached-protocol results are not mixed with this release.

## Scientific protocol

- Frozen ChatGLM3-6B-base, not the conversational checkpoint.
- Separate audio/video LSTMs and full valid temporal states; two learned queries
  per modality, projected to four language-model input tokens.
- Content-only: neither query receives the transcript-conditioned projection.
  Text still enters the language model. No semantic disentanglement is claimed.
- Mixed: only the second query per modality receives the text projection.
- Keep the unused text projection for compatibility with the controlled model's
  allocated state; report active and allocated parameter counts separately.
- AdamW, constant learning rate 5e-5, epsilon1e-4, betas(0.9,0.999), weight decay.01,
  gradient norm clipping1 after AMP unscale, batch8, FP16, no accumulation.
- Regression selects minimum validation MAE; classification selects maximum
  validation weighted F1. Use original dataset patience and output budget.
- Preserve official prompt wrapping, answer-padding supervision, no extra EOS,
  parser fallback to zero and dataset metric clipping/rounding.
- Generated validation, checkpoint replay and test use `use_cache=False`,
  recomputing the full prefix. This is not an incremental KV-cache patch.
- Test follows completed training and exact fresh-process replay of the complete
  best-validation prediction JSON. Test does not drive training or selection.

## Interpretation limits

The original experiments used three seeds (1111,2222,3333), while the reference
paper reported five. Software and hardware differ from the reference setup.
Do not claim identical reproduction of the paper environment. A reorganized
implementation requires its own real-model acceptance before claiming bitwise
reproduction of historical checkpoints; portable serialization/import paths can
also differ. CPU mock tests alone do not establish real-GPU reproducibility.

SIMS-V2 test results were inspected during method development. Do not describe
that split as an untouched confirmatory evaluation or use its results to make
unqualified generalization claims. Do not pool cached, no-cache, alternate-input,
normal-early-stop and deadline-truncated results without clearly labeling them.
No unfinished CHERMA or other experiment is represented here as a completed
three-seed result. Experimental tables should come from verified artifacts,
not numbers reconstructed from training logs or a partially written manuscript.

## Submission checklist (not a venue-compliance certificate)

This archive is prepared as an anonymous code artifact: no author accounts,
server addresses, institution paths, private data, checkpoints, Git history or
paper PDF. Upstream citations and license notices are retained; anonymity does
not justify deleting third-party copyright notices.

Before submitting to a specific CVPR year:
1. Check that year's official author and supplementary-material instructions,
   allowed archive formats, size limit, deadlines and code-link/anonymity rules.
2. Inspect README, metadata, comments, file names and any new GitHub account for
   author-identifying information. A public repository can reveal identity even
   when its files appear anonymous. Do not publish a review-stage URL without
   checking the applicable rules.
3. Ensure the paper states the model, software, seeds, selection rule, precision,
   decoding policy, hardware, early stopping and any missing/truncated runs.
4. Run the documented CPU checks and at least one real-model smoke plus separate
   reload in the intended environment. Keep dataset/model assets external.
5. Verify final metrics against raw complete prediction files. Report sample
   standard deviation only when all planned seeds of a group are complete.
6. After acceptance, add actual author/citation details and the final repository
   URL. This archive deliberately does not fabricate a DOI, arXiv ID or venue.

The current year's official website could not be retrieved during packaging;
these are conservative general practices, not a claim that all CVPR-specific
requirements have been independently verified.
