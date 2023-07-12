from autochain.tools.base import Tool
from autochain.tools.google_search.util import GoogleSearchAPIWrapper


class GoogleSearchTool(Tool):
    """Tool that has capability to query the Google Search API and get back json."""

    name = "Google Search Results JSON"
    description = (
        "A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a JSON array of the query results"
    )
    num_results: int = 4
    api_wrapper: GoogleSearchAPIWrapper

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        return str(self.api_wrapper.results(query, self.num_results))
