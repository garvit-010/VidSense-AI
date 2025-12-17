from langchain_community.tools import DuckDuckGoSearchRun


class WebSearchTool:
    def __init__(self):
        self.search = DuckDuckGoSearchRun()

    def search_web(self, query: str) -> str:
        """
        Returns raw web search text for a query
        """
        return self.search.run(query)
