"""Explicit estimator continuation through public Network operations."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import FIXED_BUDGET
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from entlearn import Network
from entlearn.scikit_adapter import EONClassifier, EONRegressor

from ._fixtures import continuation_case, continuation_frame, tensor_data

pytestmark = FIXED_BUDGET


class TestRouting:
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_unfitted_warm_start_warns_once_then_continues(self, estimator_type, mode):
        model, X, y = continuation_case(estimator_type, warm_start=mode)
        with pytest.warns(UserWarning, match=r"no fitted model.*fresh fit") as seen:
            model.fit(X, y)
        notices = [warning for warning in seen if "no fitted model" in str(warning.message)]
        assert len(notices) == 1
        assert mode in str(notices[0].message)
        assert notices[0].filename == __file__
        source = model.network_
        with warnings.catch_warnings(record=True) as later:
            warnings.simplefilter("always")
            model.set_params(max_iter=6).fit(X, y)
        assert not any("no fitted model" in str(warning.message) for warning in later)
        assert model.network_ is not source
        with pytest.warns(UserWarning, match=r"no fitted model.*fresh fit"):
            clone(model).fit(X, y)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_unfitted_warning_as_error_prevents_fitting(self, estimator_type, mode, monkeypatch):
        model, X, y = continuation_case(estimator_type, warm_start=mode)
        before = vars(model).copy()

        def unexpected_fit(*args, **kwargs):
            raise AssertionError("Network.fit must not run after a warning raised as an error")

        monkeypatch.setattr(Network, "fit", unexpected_fit)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=r".*no fitted model", category=UserWarning)
            with pytest.raises(UserWarning, match=r"no fitted model.*fresh fit"):
                model.fit(X, y)
        assert not hasattr(model, "network_")
        assert vars(model).keys() == before.keys()
        assert all(vars(model)[name] is value for name, value in before.items())

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_explicit_fresh_fit_does_not_warn_about_missing_model(self, estimator_type):
        model, X, y = continuation_case(estimator_type)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            model.fit(X, y)
        assert not any("no fitted model" in str(warning.message) for warning in seen)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_search_clones_warn_without_continuing_the_source(self, estimator_type, mode):
        model, X, y = continuation_case(estimator_type)
        model.fit(X, y)
        source = model.network_
        prediction = model.predict(X)
        model.set_params(warm_start=mode)
        search = GridSearchCV(model, {"max_iter": [2]}, cv=2, error_score="raise")
        with pytest.warns(UserWarning, match=r"no fitted model.*fresh fit") as seen:
            search.fit(X, y)
        assert any("no fitted model" in str(warning.message) for warning in seen)
        assert model.network_ is source
        np.testing.assert_array_equal(model.predict(X), prediction)

    @pytest.mark.parametrize("mode", [False, "resume", "fine_tune"])
    def test_a_second_fit_follows_the_explicit_mode(self, mode):
        model, X, y = continuation_case(EONClassifier)
        X_t, y_t = tensor_data(X, y)
        model.fit(X, y)
        source = model.network_
        # A resume reproduces the uninterrupted fit, so the fresh fit takes another seed.
        expected = {
            False: lambda: Network.fit(model.recipe, X_t, y_t, seed=8, max_iter=6, tol=0),
            "resume": lambda: source.resume(X_t, y_t, max_iter=6, tol=0),
            "fine_tune": lambda: source.fine_tune(X_t, y_t, max_iter=6, tol=0),
        }[mode]()
        model.set_params(warm_start=mode, max_iter=6, random_state=8).fit(X, y)
        assert model.loss_curve_ == expected.diagnostics.loss_history
        np.testing.assert_array_equal(model.predict_proba(X), expected.predict(X_t).cpu().numpy())

    @pytest.mark.parametrize(
        "mode",
        [True, 0, None, "automatic", np.array([1, 2])],
        ids=["true", "zero", "none", "unknown-string", "array"],
    )
    def test_invalid_mode_explains_the_available_choices(self, mode):
        model, X, y = continuation_case(EONClassifier, warm_start=mode)
        with pytest.raises(ValueError, match=r"resume.*fine_tune.*False"):
            model.fit(X, y)
        assert not hasattr(model, "network_")

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_continuation_requires_one_initialisation_but_unfitted_clones_fit_fresh(
        self, estimator_type, mode
    ):
        model, X, y = continuation_case(estimator_type, n_inits=2)
        model.fit(X, y)
        source = model.network_
        model.set_params(warm_start=mode)
        with pytest.raises(ValueError, match="n_inits=1"):
            model.fit(X, y)
        assert model.network_ is source
        copied = clone(model)
        with pytest.warns(UserWarning, match=f"warm_start='{mode}'.*no fitted model"):
            copied.fit(X, y)
        assert len(copied.network_.diagnostics.initialisation_outcomes) == 2

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("n_inits", [True, np.bool_(True), 1.0])
    def test_continuation_rejects_a_non_integer_initialisation_count(self, estimator_type, n_inits):
        model, X, y = continuation_case(estimator_type)
        model.fit(X, y)
        source = model.network_
        model.set_params(warm_start="resume", n_inits=n_inits)
        with pytest.raises(ValueError, match="n_inits=1"):
            model.fit(X, y)
        assert model.network_ is source

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_continuation_rejects_an_unnamed_feature_count_change(self, estimator_type, mode):
        model, X, y = continuation_case(estimator_type)
        model.fit(X, y)
        source = model.network_
        model.set_params(warm_start=mode)
        with pytest.raises(ValueError, match="expecting 2 features"):
            model.fit(np.column_stack([X, X[:, 0]]), y)
        assert model.network_ is source

    def test_continuation_validates_its_verbosity(self):
        model, X, y = continuation_case(EONRegressor)
        model.fit(X, y)
        source = model.network_
        model.set_params(warm_start="resume", verbose=-1)
        with pytest.raises(ValueError, match="verbose must be"):
            model.fit(X, y)
        assert model.network_ is source

    def test_continuation_refuses_out_of_range_values_without_an_overflow_warning(self):
        model, X, y = continuation_case(EONRegressor, dtype=torch.float32)
        model.fit(X, y)
        model.set_params(warm_start="resume")
        X = X.copy()
        X[0, 0] = 1e300
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(ValueError, match="infinity or a value too large"):
                model.fit(X, y)

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_recipe_changes_require_fine_tune_or_fresh_fit(self, estimator_type):
        model, X, y = continuation_case(estimator_type)
        model.fit(X, y)
        source = model.network_
        replacement = model.recipe.replace_block(model.recipe.blocks[0].name, epsilon=0.2)
        model.set_params(warm_start="resume", recipe=replacement)
        with pytest.raises(ValueError, match=r"Recipe.*fine_tune"):
            model.fit(X, y)
        assert model.network_ is source
        model.set_params(warm_start="fine_tune").fit(X, y)
        assert model.network_.recipe == replacement


class TestVocabularies:
    @pytest.mark.filterwarnings("ignore:resume checkpoint loss differs:UserWarning")
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_one_known_class_keeps_all_output_meanings(self, mode):
        model, X, y = continuation_case(EONClassifier)
        model.fit(X, y)
        classes = model.classes_.copy()
        rows = np.arange(len(y)) if mode == "resume" else np.arange(16, 32)
        labels = np.full(len(rows), "north")
        model.set_params(warm_start=mode, max_iter=6).fit(X[rows], labels)
        np.testing.assert_array_equal(model.classes_, classes)
        assert model.predict_proba(X).shape == (len(X), len(classes))

    @pytest.mark.filterwarnings("ignore:resume checkpoint loss differs:UserWarning")
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_one_known_category_keeps_the_feature_and_vocabulary(self, estimator_type, mode):
        model, X, y = continuation_case(estimator_type)
        frame = continuation_frame(X)
        model.set_params(categorical_features="from_dtype").fit(frame, y)
        source = model.network_
        rows = np.arange(len(y)) if mode == "resume" else np.arange(0, len(y), 2)
        model.set_params(warm_start=mode, max_iter=6)
        model.fit(frame.iloc[rows].assign(category="b"), y[rows])
        assert model.feature_layout_.categorical_indices == (2,)
        np.testing.assert_array_equal(model.feature_layout_.categories[0], ["a", "b"])
        assert model.network_.schema.M_cat == source.schema.M_cat

    @pytest.mark.filterwarnings("ignore:categorical column .* single level:UserWarning")
    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    @pytest.mark.parametrize("dropped", [False, True])
    def test_unknown_category_rejects_without_publishing(self, estimator_type, dropped):
        model, X, y = continuation_case(estimator_type)
        frame = continuation_frame(X, categories="a" if dropped else None)
        model.set_params(categorical_features="from_dtype").fit(frame, y)
        source = model.network_
        before = model.predict(frame)
        model.set_params(warm_start="fine_tune")
        with pytest.raises(ValueError, match=r"unseen at fit time: .*'c'"):
            model.fit(frame.assign(category="c"), y)
        assert model.network_ is source
        np.testing.assert_array_equal(model.predict(frame), before)

    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_unknown_label_rejects_even_with_the_same_number_of_classes(self, mode):
        model, X, y = continuation_case(EONClassifier)
        model.fit(X, y)
        source = model.network_
        before = model.predict(X)
        model.set_params(warm_start=mode)
        with pytest.raises(ValueError, match="unseen"):
            model.fit(X, np.where(y == "north", "south", y))
        assert model.network_ is source
        np.testing.assert_array_equal(model.predict(X), before)

    @pytest.mark.filterwarnings("ignore:resume checkpoint loss differs:UserWarning")
    @pytest.mark.parametrize("mode", ["resume", "fine_tune"])
    def test_large_integer_labels_and_unlabelled_rows_keep_exact_codes(self, mode):
        model, X, labels = continuation_case(EONClassifier)
        classes = np.array([2**53 + 1, 2**53 + 3, 2**53 + 5], dtype=np.int64)
        y = classes[np.unique(labels, return_inverse=True)[1]]
        model.fit(X, y)
        y = np.full(len(X), classes[1], dtype=np.int64)
        y[::3] = -1
        model.set_params(warm_start=mode).fit(X, y)
        np.testing.assert_array_equal(model.classes_, classes)
        with pytest.raises(ValueError, match="unseen"):
            model.fit(X, np.full(len(X), 2**53 + 2, dtype=np.int64))

    @pytest.mark.parametrize("estimator_type", [EONClassifier, EONRegressor])
    def test_column_order_and_modality_changes_fail_before_delegation(self, estimator_type):
        model, X, y = continuation_case(estimator_type)
        frame = pd.DataFrame(X, columns=["x", "z"])
        model.fit(frame, y)
        source = model.network_
        model.set_params(warm_start="fine_tune")
        with pytest.raises(ValueError, match="feature names"):
            model.fit(frame[["z", "x"]], y)
        model.set_params(categorical_features=[0])
        with pytest.raises(ValueError, match="column meanings"):
            model.fit(frame, y)
        assert model.network_ is source
