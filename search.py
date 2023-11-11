import os
import json
import requests
import datetime as dt
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

UTC_TIMESTAMP = int(dt.datetime.utcnow().timestamp())

# Generic settings
TIMEOUT = 60
RETRIES = 3
DELAY = 5
BACKOFF = 2
N_RESULTS = 4

def search_bing(kwargs) -> list[str]:
    """Helper function to search Bing for relevant documents"""
    query = kwargs.get("query", None)
    search_index = kwargs.get("search_index", None)
    if query is None or search_index is None:
        return []
    
    bing_search_endpoint = os.getenv("BING_SEARCH_ENDPOINT", "").rstrip("/")
    bing_search_key = os.getenv("BING_CUSTOMSEARCH_KEY", "")
    url = f"{bing_search_endpoint}/v7.0/custom/search"
    
    match search_index:
        case "DEFAULT":
            url = f"{bing_search_endpoint}/v7.0/search"
            bing_search_key = os.getenv("BING_SEARCH_KEY", "")
            params = {"q": query, "count": N_RESULTS}
        case _:
            return ["Invalid search_index parameter. Valid choices is 'DEFAULT'."]

    headers = {"Ocp-Apim-Subscription-Key": bing_search_key}
    res = call_API("GET", url, headers=headers, params=params)
    custom_res = res.get("webPages", {}).get("value", [])

    # if DEBUG:
    #     with st.sidebar:
    #         st.write("Bing Search Query")
    #         st.text(f"Query: {query}")
    #         st.text(f"Search_index: {search_index}")
    #         st.write("Bing Search Results")
    #         st.json(custom_res, expanded=False)

    outputs = []

    if len(custom_res) > 0:
        for item in custom_res:
            outputs.append(
                # f"{item['name']}\n\n{item['snippet']}\n\nSource: {item['url']}")
                f"{item['snippet']}\n\nSource: {item['url']}")
    else:
        outputs.append(f"No relevant results found on {search_index} search index.")
    
    return outputs

@retry(stop=stop_after_attempt(RETRIES), wait=wait_exponential(multiplier=BACKOFF, min=DELAY), reraise=True, retry_error_callback=logger.error)
def call_API(
    method: str,
    url: str,
    params: dict = {},
    headers: dict = {},
    data: dict = {},
) -> dict:
    logger.debug(f"Requesting {method} {url} with params {params}")
    if data:
        logger.debug(f"Request data: {data}")

    response = requests.request(method, url, params=params, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    elif response.status_code in [400, 401, 403, 404, 429, 500, 503]:
        return response.json()
    else:
        raise Exception(f"Request failed with status {response.status_code}: {response.reason}, {response.text}")   