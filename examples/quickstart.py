"""Architecture and physics smoke test requiring no external data."""

import torch

from samg.models import DirectMuellerVAE, FourIncidentPIVAE, OperatorPIVAE
from samg.physics import minimum_cloude_eigenvalue, stokes_is_valid


def main() -> None:
    generator = torch.Generator().manual_seed(2044)
    batch = 8
    patch = torch.randn(batch, 16, 5, 5, generator=generator)
    matrix = patch[:, :, 2, 2].reshape(batch, 4, 4)
    position = torch.rand(batch, 2, generator=generator)
    spectral = 5.0 * torch.rand(batch, 1, generator=generator)

    for model in (FourIncidentPIVAE(), OperatorPIVAE(), DirectMuellerVAE()):
        model.eval()
        with torch.no_grad():
            output = model(patch, matrix, position, spectral, sample_latent=False)
        minimum = minimum_cloude_eigenvalue(output["m_hat"])
        line = {
            "model": type(model).__name__,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "output_shape": list(output["m_hat"].shape),
            "mean_lambda_min": float(minimum.mean()),
        }
        if "s_out" in output:
            line["valid_stokes_fraction"] = float(stokes_is_valid(output["s_out"]).float().mean())
        print(line)


if __name__ == "__main__":
    main()
