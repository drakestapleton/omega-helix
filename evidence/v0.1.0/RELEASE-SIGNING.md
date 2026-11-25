# Omega Helix `v0.1.0` Pure-Omega RFC 8032 Ed25519 Release Signing

Release artifacts (`omega-helix-v0.1.0-SHA256SUMS.txt`) are signed with a persistent Ed25519 key generated and executed purely in seeded Omega (`.om`) using `sha512.om`, `ed25519.om`, `ed_sign.om`, and `sign_ddc.om` from `omega-mv`, cross-verified against Python `cryptography`'s RFC 8032 `Ed25519PublicKey`.

- **Public Key (32-byte Ed25519, hex):** `f78a47060b1e18cbde72cfb920301ed464858f820ec43489414be6367d4296c4`
- **Public Key File:** [`release-pubkey.ed25519.hex`](release-pubkey.ed25519.hex)

## Independent Offline Verification

Download the release assets (`omega-helix-v0.1.0-source.tar.gz`, `omega-helix-v0.1.0-SHA256SUMS.txt`, `omega-helix-v0.1.0-SHA256SUMS.txt.ed25519.sig`, and `omega-helix-v0.1.0-release-pubkey.ed25519.hex`), then verify both the SHA-256 digests and the raw RFC 8032 Ed25519 signature:

```sh
sha256sum -c omega-helix-v0.1.0-SHA256SUMS.txt   # or shasum -a 256 -c ...

python3 - <<'PY'
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
pk = bytes.fromhex(open("omega-helix-v0.1.0-release-pubkey.ed25519.hex").read().strip())
sig = bytes.fromhex(open("omega-helix-v0.1.0-SHA256SUMS.txt.ed25519.sig").read().strip())
msg = open("omega-helix-v0.1.0-SHA256SUMS.txt", "rb").read()
Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg)
print("ED25519_RELEASE_SIGNATURE_VALID")
PY
```
