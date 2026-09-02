"""scENVI — ENVI and COVET.

``ENVI`` is resolved lazily via ``__getattr__`` (PEP 562), so importing scenvi does
not import jax, flax, optax, clu or tensorflow_probability. COVET is pure
numpy/sklearn/scanpy and uses none of them, and resolving ENVI on first use keeps a
breakage anywhere in that stack from taking ``compute_covet`` down with it — which
is what #9 was, and what the tensorflow_probability pin does today.

``from scenvi import ENVI`` behaves exactly as before.
"""

from scenvi.utils import compute_covet  # noqa: F401

__all__ = ["ENVI", "compute_covet"]


def __getattr__(name):
    """Resolve ``ENVI`` on first access, so importing scenvi stays free of jax."""
    if name == "ENVI":
        from scenvi._envi import ENVI

        return ENVI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Keep ``ENVI`` discoverable despite the lazy import."""
    return sorted(__all__)
