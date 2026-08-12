"""compute_covet's preprocessing decisions are explicit and consistent."""

import warnings

import anndata
import numpy as np
import pytest

import scenvi

K = 6


@pytest.fixture
def counts_data():
    """Raw counts: non-negative, so the historical heuristic log-transforms them."""
    rng = np.random.default_rng(0)
    return anndata.AnnData(
        X=rng.poisson(5.0, size=(64, 8)).astype(np.float64),
        obsm={"spatial": rng.normal(size=(64, 2))},
    )


@pytest.fixture
def batched_data():
    rng = np.random.default_rng(1)
    adata = anndata.AnnData(
        X=rng.poisson(5.0, size=(64, 8)).astype(np.float64),
        obsm={"spatial": rng.normal(size=(64, 2))},
    )
    adata.obs["sample"] = ["a"] * 32 + ["b"] * 32
    return adata


class TestLogTransform:
    def test_heuristic_warns_when_it_log_transforms(self, counts_data):
        with pytest.warns(UserWarning, match="log_transform=False"):
            scenvi.compute_covet(counts_data, k=K, g=8, batch_key=-1)

    @pytest.mark.parametrize("log_transform", [True, False])
    def test_explicit_choice_does_not_warn(self, counts_data, log_transform):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            scenvi.compute_covet(counts_data, k=K, g=8, batch_key=-1, log_transform=log_transform)

    def test_log_transform_false_matches_pre_logged_input(self, counts_data):
        """log_transform=False must use X verbatim, i.e. agree with logging it ourselves."""
        expected, _, _ = scenvi.compute_covet(counts_data, k=K, g=8, batch_key=-1, log_transform=True)

        logged = counts_data.copy()
        logged.X = np.log(counts_data.X + 1)
        actual, _, _ = scenvi.compute_covet(logged, k=K, g=8, batch_key=-1, log_transform=False)

        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_default_is_unchanged_for_existing_users(self, counts_data):
        """The heuristic still fires; this PR warns about it, it does not alter it."""
        with pytest.warns(UserWarning):
            default, _, _ = scenvi.compute_covet(counts_data, k=K, g=8, batch_key=-1)
        explicit, _, _ = scenvi.compute_covet(counts_data, k=K, g=8, batch_key=-1, log_transform=True)

        np.testing.assert_allclose(default, explicit, rtol=1e-10, atol=1e-12)

    def test_use_obsm_is_taken_at_face_value(self, counts_data):
        counts_data.obsm["feat"] = np.asarray(counts_data.X[:, :4], dtype=np.float64)
        covet, _, _ = scenvi.compute_covet(counts_data, k=K, use_obsm="feat", batch_key=-1)

        logged = counts_data.copy()
        logged.obsm["feat"] = np.log(counts_data.obsm["feat"] + 1)
        covet_logged, _, _ = scenvi.compute_covet(logged, k=K, use_obsm="feat", batch_key=-1)

        assert not np.allclose(covet, covet_logged)


class TestBatchKey:
    def test_missing_explicit_batch_key_raises(self, batched_data):
        with pytest.raises(ValueError, match="not a column of spatial_data.obs"):
            scenvi.compute_covet(batched_data, k=K, g=8, batch_key="smaple", log_transform=False)

    def test_default_batch_key_still_falls_back(self, counts_data):
        """`batch_key='batch'` with no such column keeps meaning "no batches"."""
        covet, _, _ = scenvi.compute_covet(counts_data, k=K, g=8, log_transform=False)
        assert covet.shape == (counts_data.n_obs, 8, 8)

    def test_explicit_minus_one_is_accepted(self, batched_data):
        covet, _, _ = scenvi.compute_covet(batched_data, k=K, g=8, batch_key=-1, log_transform=False)
        assert covet.shape == (batched_data.n_obs, 8, 8)

    def test_batches_change_the_result(self, batched_data):
        pooled, _, _ = scenvi.compute_covet(batched_data, k=K, g=8, batch_key=-1, log_transform=False)
        per_batch, _, _ = scenvi.compute_covet(batched_data, k=K, g=8, batch_key="sample", log_transform=False)
        assert not np.allclose(pooled, per_batch)


class TestBatchSize:
    def test_batch_size_does_not_change_the_result(self, counts_data):
        """batch_size is a memory knob and must be numerically inert.

        Both results are cast to float32 on return, so the dtype difference is
        invisible -- but the batched path used to accumulate in float32 before the
        regularization term and the square root were computed, which moved COVET by
        ~7e-8 relative and COVET_SQRT by ~6e-6.
        """
        whole, sqrt_whole, _ = scenvi.compute_covet(
            counts_data, k=K, g=8, batch_key=-1, log_transform=False
        )
        chunked, sqrt_chunked, _ = scenvi.compute_covet(
            counts_data, k=K, g=8, batch_key=-1, log_transform=False, batch_size=16
        )

        np.testing.assert_allclose(chunked, whole, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(sqrt_chunked, sqrt_whole, rtol=1e-10, atol=1e-12)

