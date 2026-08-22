# Model card

## Intended use

SAMG is intended for research on Mueller polarimetry, Stokes-conditioned
operator reconstruction, latent-field generation and representation transfer.
It can serve as a preconditioner for downstream segmentation or classification
experiments when those tasks have their own leakage-safe protocol.

## Inputs and outputs

The core model consumes a local 16-channel Mueller patch, its central 4 x 4
matrix, normalized position and a dimensionless spectral condition. It returns
latent distribution parameters, a latent sample, incident and emerging Stokes
banks, and a ridge-fitted Mueller reconstruction.

## Guarantees and non-guarantees

Emerging Stokes rays are admissible by construction up to floating-point
tolerance. Global Cloude-PSD realizability is measured separately and is not
guaranteed by a finite ray bank. The optional PSD projection is a post-processing
baseline, not part of the SAMG forward pass.

## Limitations

- The historical four-group model depends on minibatch composition.
- The scalar spectral interface does not by itself distinguish wavelength,
  spatial frequency and categorical bands.
- Dataset-specific generalization, calibration and downstream utility cannot be
  inferred from the synthetic smoke test.
- Biomedical datasets may have limited demographic, geographic, instrument and
  acquisition-protocol diversity.
- Generated samples may reproduce training examples; memorization checks are
  required before any release of samples or weights.

## Safety and societal considerations

The software is not a clinical device and must not be used for diagnosis or
treatment decisions. False confidence in physical admissibility or domain
transfer could propagate errors to biomedical predictions. Human oversight,
external validation, calibration and explicit failure analysis are required
for any high-stakes study.

## Data and weights

No data or pretrained weights are distributed. Their licences, consent terms,
privacy constraints and checksums must be documented independently.
