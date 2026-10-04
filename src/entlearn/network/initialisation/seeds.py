"""Root validation and deterministic candidate seed planning."""

from numbers import Integral

import torch

from entlearn._seeds import _SEED_UPPER_BOUND


def _validated_seed(seed: object) -> int:
    """Return a non-boolean integral root seed as an int in the supported range."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= int(seed) < _SEED_UPPER_BOUND
    ):
        raise ValueError("seed must be an integer in [0, 2**63)")
    return int(seed)


def _plan_seeds(seed: int, n_inits: int, device: torch.device) -> tuple[int, ...]:
    """Plan Network's ordered candidate seeds without touching the global RNG.

    One candidate uses the root; several draw sub-seeds from a generator on ``device``.
    """
    if n_inits == 1:
        return (seed,)
    generator = torch.Generator(device=device).manual_seed(seed)
    return tuple(
        int(value)
        for value in torch.randint(
            high=_SEED_UPPER_BOUND - 1,
            size=(n_inits,),
            generator=generator,
            device=generator.device,
        )
    )
