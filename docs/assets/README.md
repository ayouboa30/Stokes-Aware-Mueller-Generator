# Figure assets

These files are fixed scientific figures rendered from archived experimental
outputs. They contain no checkpoint, latent tensor, patient identifier or local
filesystem path. The underlying research datasets and model weights are not
distributed by this repository.

| Asset | Scientific content | Evaluation status |
|---|---|---|
| `wm2_matched_results.png` | Matched SAMG/direct-VAE training curves and held-out contrasts | Held-out internal evaluation |
| `corpus_summary.png` | Aggregate Mueller signatures and sample counts for five sources | Training-corpus summary |
| `latent_field_statistics.png` | Autocorrelation, marginal statistics and variogram of the unified latent field | Aggregate latent analysis |
| `matrix_reconstruction_example.png` | One local target, three reconstructions, errors and Cloude spectra | Held-out internal example |
| `generative_comparison.png` | Generated depolarization, diattenuation and polarizance fields | Archived synthetic samples |
| `generated_mueller_fields.png` | Complete 16-channel generated Mueller fields | Archived generated samples |
| `stokes_interrogation.png` | One generated local operator queried by four incident Stokes states | Deterministic physical calculation |
| `synthetic_cervix_field.png` | Generated cervical depolarization channel | Training-only synthetic output |
| `synthetic_cervix_mask.png` | Generated multiclass mask paired with the cervical field | Training-only synthetic output |
| `colopola_flow_augmentation.png` | Few-shot AUC with and without Flow Matching augmentation | Root-grouped held-out evaluation |

Human-image plates are excluded while redistribution permissions for the source
datasets are being checked. The remaining assets are aggregate statistics,
local matrix heatmaps or generated outputs and do not grant redistribution
rights to any underlying dataset.
