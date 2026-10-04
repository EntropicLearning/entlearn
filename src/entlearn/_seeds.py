"""Shared integer seed bounds and connection identity, independent of tensor RNGs."""

import hashlib

_SEED_UPPER_BOUND = 1 << 63
_CONNECTION_SEED_DIGEST_BYTES = 8


def _connection_sub_seed(root_seed: int, name: str) -> int:
    """Return a seed depending only on the root and the stable connection name."""
    digest = hashlib.blake2b(
        f"{root_seed}:{name}".encode(), digest_size=_CONNECTION_SEED_DIGEST_BYTES
    ).digest()
    return int.from_bytes(digest, "little") >> 1
