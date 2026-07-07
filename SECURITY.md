# Security Policy

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

Please report security vulnerabilities via email to **security@szlholdings.com** with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact assessment
4. Any suggested mitigations

### Response SLA

| Severity | Initial Response | Resolution Target |
|---|---|---|
| Critical | 24 hours | 7 days |
| High | 48 hours | 30 days |
| Medium | 5 business days | 90 days |
| Low | 10 business days | 180 days |

We follow a **90-day responsible disclosure** policy.

## Supply-Chain Security

- **SLSA Build Level 1** — build provenance generated per release (honest; not L2/L3)
- **DCO required** — all commits carry `Signed-off-by:` trailers per [Linux Foundation DCO](https://developercertificate.org/)
- **Cosign keyless signing** — verify with `cosign verify ghcr.io/szl-holdings/szl-kernels:<tag>`
- **SBOM** — CycloneDX SBOM attached to each release

## Contact

- **Security disclosures:** security@szlholdings.com
- **General:** hello@szlholdings.com
- **Website:** https://szlholdings.com

*This policy follows the [OpenSSF Vulnerability Disclosure Guide](https://github.com/ossf/oss-vulnerability-guide).*
