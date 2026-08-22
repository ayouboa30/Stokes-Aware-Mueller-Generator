# Compatibility with research checkpoints

This release contains no weights. For authorized checkpoints from the research
workspace, load only a state dictionary into the corresponding known class.

## Core models

- `FourIncidentPIVAE` preserves the historical module names, tensor shapes and
  parameter count (103,712 at latent dimension 8).
- `DirectMuellerVAE` preserves the clean direct-baseline state layout and
  parameter count (109,760 at latent dimension 8).
- `OperatorPIVAE` preserves the architecture state layout of the
  incident-independent CPU variant (388,517 parameters at default settings).

The public `OperatorPIVAE` uses the mathematically oriented ridge expression
`Y S^T (S S^T + lambda I)^-1`. An earlier research implementation multiplied
the non-symmetric tetrahedral bank in the opposite order. Its neural weights
load strictly, but its reconstructed Mueller output is intentionally corrected
and should be reevaluated rather than assumed numerically identical.

## Latent generators

At their default eight-channel dimensions, PixelCNN, PixelRNN, latent VDM and
Flow Matching retain the original parameter names and sizes. The public
`MaskedConv2d` applies its mask functionally instead of mutating
`weight.data`; this removes state mutation while preserving the masked
convolution.

## Safe loading

PyTorch checkpoint files can execute arbitrary code when loaded unsafely. Use
`weights_only=True`, verify an external SHA-256 checksum and reject unexpected
or missing keys:

```python
import torch
from samg.models import FourIncidentPIVAE

payload = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
state = payload.get("model_state", payload.get("model", payload))
model = FourIncidentPIVAE(latent_dim=8)
model.load_state_dict(state, strict=True)
```

Do not add the checkpoint to Git. Store its licence, training-data scope,
architecture name, preprocessing contract and checksum alongside the external
release asset.

Some legacy checkpoints contain version objects and cannot be opened with
`weights_only=True`. For a file whose source and checksum have already been
verified, `tools/convert_trusted_checkpoint.py` performs one explicitly gated
legacy load and emits a tensor-only payload. Never use that conversion on an
untrusted download.

The research variant historically named `pivae_tetra` is not
`OperatorPIVAE`: its encoder and decoder dimensions differ. Only checkpoints
created by the `CPUFourIncidentPIVAE` architecture map to `OperatorPIVAE`.
