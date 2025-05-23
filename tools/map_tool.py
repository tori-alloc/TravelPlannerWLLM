from langchain.tools import tool
import requests
import os
from dotenv import load_dotenv

load_dotenv()
KAKAO_API_KEY= os.environ.get("KAKAO_API_KEY")
GOOGLE_OAUTH_CLIENT_ID= os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
GOOGLE_OAUTH_CLIENT_SECRET= os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")

@tool
def search_kakao_places(
    query: str,
    location: str= "",
    radius: int= 10000,
    size: int= 5
) -> list[str]:
    """
    Kakao Map API를 이용해서 특정 키워드와 위치 기준으로 장소를 검색합니다.

    Args:
        query (str): _description_
        location (str, optional): _description_. Defaults to "".
        radius (int, optional): _description_. Defaults to 10000.
        size (int, optional): _description_. Defaults to 5.

    Returns:
        list[str]: _description_
    """
    headers= {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }
    params= {
        "query": query
    }
    if location: 
        lat, lng= location.split(",")
        params.update(
            {
                "x": lng,
                "y": lat,
                "radius": radius,
                "sort": "distance"
            }
        )
    res= requests.get(
        "https://dapi.kakao.com/v2/local/search/keyword.json",
        headers= headers,
        params= params
    )
    data= res.json()
    
    if "documents" not in data or not data["documents"]:
        return ["empty"]
    return [
        f"{doc['place_name']} - {doc.get('road_address_name', doc['address_name'])}"
        for doc in data["documents"]
    ]