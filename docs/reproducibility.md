# Reproducibility guide

## Separation of selection and testing

`samg-train` accepts only `--train` and `--validation`. It cannot load a test
file. `samg-evaluate` is a separate, explicit action and records whether the
chosen split is named `test`.

The training command saves four checkpoints:

- `best_mse.pt`: minimum normalized validation MSE;
- `best_physical.pt`: maximum validation fraction above the configured Cloude
  margin, with MSE as a tie-break;
- `best_compromise.pt`: minimum declared fidelity/physicality score;
- `last.pt`: last monitored epoch.

The history is written every `monitor_every` epochs rather than serializing a
checkpoint at every epoch.

Two Cloude thresholds are kept distinct. `psd_numerical_tolerance` only absorbs
floating-point error around zero. `psd_minimum_eigenvalue` is an explicit
positive margin such as `0.005` or `0.01` when a corpus-quality rule requires
one. Both values are stored in the run and evaluation protocols.

## Optimization regimes

- `samg`: fixed penalty coefficients;
- `samg-random`: every regularizer receives an independent draw from
  `Uniform(0,1)` once per epoch; reconstruction is never randomized;
- `operator`: deterministic tetrahedral probe and incident-independent latent;
- `direct`: direct Mueller VAE.

The default schedule begins with reconstruction, ramps the KL term, then
switches to channel-normalized matrix reconstruction while ramping Stokes
conditioning and Poincare penalties. Every coefficient and random draw is
stored in `history.jsonl`.

The condition-number and Poincare terms can span several orders of magnitude.
Each is divided by a robust reference estimated from per-sample medians during
the twenty epochs preceding its ramp. The fitted references are recorded in
the history and checkpoints; KL retains its conventional unnormalized scale.

## Fair comparisons

Use the same unit split, training-only statistics, latent dimension, optimizer,
number of updates, batch policy and selection rule. Report parameter counts
rather than assuming the historical SAMG and direct VAE have identical
capacity.

Optimization seeds are repeated fits, not independent biological units.
Aggregate uncertainty at the biological-unit level whenever the scientific
claim concerns generalization across patients or specimens.

## Hardware

AMP is optional. `fp16` and `bf16` are enabled only on CUDA; CPU execution uses
float32. A CUDA-specific PyTorch wheel should be installed following the
official PyTorch instructions for the target system.
