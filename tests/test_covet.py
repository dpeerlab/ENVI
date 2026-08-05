"""COVET does not depend on ENVI's deep-learning stack.

COVET is pure numpy/sklearn/scanpy. Keeping it that way is what lets
``compute_covet`` survive a breakage in jax/flax/optax/clu/tensorflow_probability
instead of being taken down with it, so the separation is asserted here rather
than left to convention.
"""

import subprocess
import sys
import textwrap

import anndata
import numpy as np
import pytest

import scenvi

#: Modules the COVET path must never pull in.
DEEP_LEARNING_STACK = ("jax", "flax", "optax", "clu", "tensorflow_probability")


@pytest.fixture
def spatial_data():
    rng = np.random.default_rng(0)
    return anndata.AnnData(
        X=rng.uniform(low=0, high=100, size=(64, 8)),
        obsm={"spatial": rng.normal(size=(64, 2))},
    )


def test_compute_covet_runs(spatial_data):
    covet, covet_sqrt, cov_genes = scenvi.compute_covet(spatial_data, k=6, g=8, batch_key=-1)

    n_cells, n_genes = spatial_data.shape
    assert covet.shape == (n_cells, n_genes, n_genes)
    assert covet_sqrt.shape == (n_cells, n_genes, n_genes)
    assert len(cov_genes) == n_genes
    assert np.isfinite(covet).all()
    assert np.isfinite(covet_sqrt).all()


def test_covet_matrices_are_symmetric_psd(spatial_data):
    covet, covet_sqrt, _ = scenvi.compute_covet(spatial_data, k=6, g=8, batch_key=-1)

    np.testing.assert_allclose(covet, covet.transpose(0, 2, 1), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(covet_sqrt, covet_sqrt.transpose(0, 2, 1), rtol=1e-6, atol=1e-8)
    # The square root is the defining property, and it is what ENVI's OT loss consumes.
    np.testing.assert_allclose(covet_sqrt @ covet_sqrt, covet, rtol=1e-5, atol=1e-6)
    assert np.linalg.eigvalsh(covet).min() > -1e-8


def test_covet_path_does_not_import_the_deep_learning_stack():
    """Running COVET end to end must leave jax and friends unimported.

    In a subprocess, because the check is on ``sys.modules`` and this session has
    almost certainly imported ENVI already.
    """
    script = textwrap.dedent(
        f"""
        import sys

        import anndata
        import numpy as np

        import scenvi

        rng = np.random.default_rng(0)
        spatial_data = anndata.AnnData(
            X=rng.uniform(low=0, high=100, size=(64, 8)),
            obsm={{"spatial": rng.normal(size=(64, 2))}},
        )
        scenvi.compute_covet(spatial_data, k=6, g=8, batch_key=-1)

        imported = [name for name in {DEEP_LEARNING_STACK!r} if name in sys.modules]
        assert not imported, "the COVET path imported " + ", ".join(imported)
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_envi_attribute_is_stable_across_repeated_access():
    """``scenvi.ENVI`` must be the class every time, never the module of the same name."""
    assert isinstance(scenvi.ENVI, type)
    assert scenvi.ENVI is scenvi.ENVI
