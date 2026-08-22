# Security and privacy

Do not open a public issue containing patient, specimen, acquisition or
credential information. Contact the repository owner privately for a security
or privacy report.

For a GitHub-hosted release, use the repository's **Security > Report a
vulnerability** private-reporting form. If private reporting is unavailable,
do not post sensitive details publicly; contact the owner through the profile
linked from the repository.

The repository does not distribute executable checkpoints. Treat third-party
PyTorch checkpoint files as untrusted: prefer `torch.load(...,
weights_only=True)`, verify a published checksum and load only a state
dictionary into a known architecture.
