import os
from unittest import mock

import pytest

from autochain.tools.google_search.util import GoogleSearchAPIWrapper


@pytest.fixture
def google_search_fixture():
    with mock.patch(
        "autochain.tools.google_search.util.GoogleSearchAPIWrapper._google_search_results",
        return_value=[{"snippet": "Barack Hussein Obama II"}],
    ):
        yield


def test_google_search(google_search_fixture) -> None:
    """Test that call gives the correct answer."""
    os.environ["GOOGLE_API_KEY"] = "mock_api_key"
    os.environ["GOOGLE_CSE_ID"] = "mock_cse_id"
    search = GoogleSearchAPIWrapper()
    output = search.run("What was Obama's first name?")
    assert "Barack Hussein Obama II" in output
