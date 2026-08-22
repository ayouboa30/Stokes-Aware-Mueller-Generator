# Architecture

## Three representations

SAMG separates three objects that should not be conflated:

1. `z`, the coordinates returned by the variational encoder;
2. the decoded action on a chosen incident bank,
   \(\Phi_B(z)=\widehat S_{\mathrm{out}}\);
3. the regularized operator fitted to that action,
   \(\Psi_B(z)=\widehat M\).

For incident states stored as columns in \(S\) and decoded responses stored in
\(Y\), the fitted operator is

\[
\widehat M = YS^\top(SS^\top + \lambda I)^{-1}.
\]

With a full-rank bank, the action error and operator error induce equivalent
Frobenius norms:

\[
\sigma_{\min}(S)\lVert M-\widehat M\rVert_F
\leq \lVert (M-\widehat M)S\rVert_F
\leq \sigma_{\max}(S)\lVert M-\widehat M\rVert_F.
\]

`samg.physics.probe_norm_bounds` exposes these two constants.

For several banks, `samg.operator` fits one shared operator, measures the
normalized closure residual `C_op`, compares groupwise fits with the shared fit
and supplies an explicit linearity loss. Held-out incident states can be scored
without refitting the operator.

## Historical SAMG

`ConditionalCNNPIVAE` encodes a 5 x 5 Mueller neighborhood, its central
matrix, position, dimensionless spectral condition and a learned incident
bank. Its decoder predicts four admissible emerging Stokes states and the
ridge step reconstructs a Mueller matrix.

`FourIncidentPIVAE` reproduces the historical training semantics: a minibatch
is split into four contiguous groups and one learned bank is generated from
the mean spectral condition of each group. This makes the bank dependent on
batch composition. Reproduction therefore requires the same grouping policy.

## Per-observation operator variant

`OperatorPIVAE` uses a deterministic tetrahedral bank for every observation,
each sample's own spectral condition and an encoder that does not receive
`S_in`. This removes batch-context dependence and makes `z` independent of the
particular interrogation bank. It is a separate architecture, not a silent
change to the historical model.

## Direct VAE baseline

`DirectMuellerVAE` has the same inputs and latent dimension but decodes the
sixteen Mueller coefficients directly. It is the principal architecture-level
baseline. Parameter counts are reported for every run; the historical models
are not described as exactly capacity matched.

## Physical checks

The decoder constrains each emerging Stokes state to the forward Lorentz cone.
This local ray validity is distinct from global Cloude realizability of the
fitted operator. `samg.physics` therefore evaluates both independently.

The default PSD projection clips negative eigenvalues of `H(M)`. It projects
onto the full PSD cone and does not preserve `M00` unless the caller explicitly
requests post-projection renormalization.

## Latent-field generators

PixelCNN and PixelRNN factorize the latent field autoregressively. The latent
VDM predicts noise along a cosine diffusion schedule. Flow Matching learns the
velocity of the linear path between Gaussian noise and data. VDM and Flow
Matching share the same backbone and default width, so their neural capacities
are equal at fixed settings.

For downstream image tasks, `samg.tasks.polarimetric_d4` applies the eight
interpolation-free spatial symmetries together with their Mueller reference-frame
change. The accompanying class mask receives the same spatial transform.
