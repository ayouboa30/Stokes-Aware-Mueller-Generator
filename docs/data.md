# Data contract

The original optical and biomedical datasets are not distributed here.

Each `.pt` split follows schema version 1 and contains:

| Field | Shape | Meaning |
|---|---:|---|
| `patch` | `N x 16 x 5 x 5` | local Mueller neighborhood |
| `target` | `N x 4 x 4` | central Mueller matrix |
| `position` | `N x 2` | coordinates normalized to `[0,1]` from train only |
| `spectral` | `N x 1` | dimensionless condition on the historical `[0,5]` scale |
| `source_index` | `N` | anonymous source-domain index |
| `record_index` | `N` | anonymous within-release record index |
| `unit_keys` | `N` strings | stable anonymous `source:unit` keys |
| `split` | scalar string | `train`, `validation` or `test` |

`patch` and `target` are in physical Mueller space, not per-channel
standardized diffusion space. Cloude transforms and physical metrics must be
computed only after any normalization has been inverted.

Raw wavelengths in nanometres, spatial frequencies and categorical band
indices must not be placed in the same scalar without an explicit upstream
typing and normalization step. This release preserves the historical scalar
model interface; a multisource adapter should document its typed encoding.

## Split safety

`unit_keys` must be stable across all partitions. Split-local integer patient
indices are invalid because equal integers in different files can denote
different units, while one real unit can receive different local integers.
`validate_disjoint_units` rejects any repeated stable key across partitions.

The keys must be anonymized and must not contain names, hospital codes,
acquisition dates or original file paths. The release checker blocks common
sensitive patterns, but anonymization remains the data controller's
responsibility.

## Normalization

Fit all statistics on training data only. `MuellerNormalizer.fit` and
`channel_scale` follow this rule. Persist the fitted values in a private run
manifest and apply them unchanged to validation and test data.

## Synthetic fixture

`samg-make-synthetic` creates PSD Mueller matrices and anonymous disjoint unit
keys. It is designed solely to test the software path; it is not a benchmark
and must not be mixed with reported experimental results.

## Latent-field split

Generator inputs use a separate schema with `kind = samg_latent_fields`, a
tensor `fields` of shape `N x C x H x W`, a split name and one stable anonymous
`unit_key` per field. `samg-train-generator` rejects unit overlap between train
and validation and standardizes every channel using train-only statistics.
