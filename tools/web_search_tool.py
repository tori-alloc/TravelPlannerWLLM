from langchain.tools import tool

@tool
def web_search(query:str) -> list[str]:
    """
    Google 검색 결과를 반환합니다.

    Args:
        query (str): _description_

    Returns:
        list[str]: _description_
    """
    return [
        f"['{query}' 검색 결과 보기](https://www.google.com/search?q={query})"
    ]