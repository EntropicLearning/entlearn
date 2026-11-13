"""Check the conftest guard that fails a match= spelling out a whole message."""

import warnings

import pytest

MESSAGE = "one two three four five six"
GUARD = "spells out the whole message"


class TestWholeMessageGuard:
    def test_a_raises_match_spelling_out_the_whole_message_fails(self):
        with (
            pytest.raises(pytest.fail.Exception, match=GUARD),
            pytest.raises(ValueError, match=MESSAGE),
        ):
            raise ValueError(MESSAGE)

    def test_a_warns_match_spelling_out_the_whole_message_fails(self):
        with (
            pytest.raises(pytest.fail.Exception, match=GUARD),
            pytest.warns(UserWarning, match=MESSAGE),
        ):
            warnings.warn(MESSAGE, UserWarning, stacklevel=1)

    def test_a_short_phrase_of_the_message_passes(self):
        with pytest.raises(ValueError, match="four five"):
            raise ValueError(MESSAGE)
