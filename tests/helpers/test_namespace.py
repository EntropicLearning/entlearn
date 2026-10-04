"""The helper namespace loads its members on attribute access."""

import pytest


class TestHelperNamespace:
    def test_optional_helper_names_resolve_lazily_and_unknown_ones_are_named(self, monkeypatch):
        from entlearn import helpers

        # An earlier import may have bound the submodule, hiding the lazy resolution.
        lazy = "reporting"
        monkeypatch.delattr(helpers, lazy, raising=False)
        assert getattr(helpers, lazy).__name__ == "entlearn.helpers.reporting"
        absent = "not_a_helper"
        with pytest.raises(AttributeError, match="has no attribute 'not_a_helper'"):
            getattr(helpers, absent)
