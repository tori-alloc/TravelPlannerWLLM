import os
from dotenv import load_dotenv
import requests

from langchain.tools import Tool

load_dotenv()

KAKAO_API_KEY = os.environ.get("KAKAO_API_KEY")

def search_place(query : str) -> list:
    headers = { "Authorization" : f"KakaoAK {KAKAO_API_KEY}" }
    params = { "query" : query }
    response = requests.get(
        "https://dapi.kakao.com/v2/local/search/keyword.json",
        headers = headers,
        params = params
    )
    
    if response.status_code != 200:
        raise ValueError("잠시 후 다시 시도해주세요.")
    
    data= response.json()
    
    print()
    print()
    print(data)
    print()
    print()
    if not data.get("documents"):
        raise ValueError("검색 결과가 없습니다.")
    places = [
        f"{doc['place_name']} - {doc['address_name']}"
        for doc in data['documents']
    ]
    # return "\n".join(places)
    return places

place_search_tool = Tool(
    name= "PlaceSearch",
    description= "여행 장소를 검색하는 도구입니다.",
    func= search_place
)