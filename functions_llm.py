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
    

functions = [
    {
        "name": "search_bing",
        "description": "Search for relevant hits from Bing.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"Query string to search the posts for, inferred from the user message and ALWAYS translated to ENGLISH before passing on to the function. Sample query styles: 'expected development of stainless steel market pricing in December 2022' or 'possible energy price developments over January-March 2023'. Note: today is {dt.datetime.utcfromtimestamp(UTC_TIMESTAMP).strftime('%Y-%m-%d')}."
                },
                "search_index": {
                    "type": "string",
                    "description": "Name of the Bing Search index to use. Valid choices: 'DEFAULT'. If 'DEFAULT' is used, the search will be performed on the entire web."
                }
            },
            "required": ["query", "search_index"]
        }
    }
]

available_functions = {
    "search_bing": search_bing,
}

# Snippet to call OpenAI with functions

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=TIMEOUT,
        function_call="auto",
        functions=functions,
    )
    first_choice_message = response["choices"][0]["message"]
    if first_choice_message["content"] is not None:
        reply_text = first_choice_message["content"].strip()
        function_call = None
    else:
        reply_text = ""
        function_call = first_choice_message["function_call"]

    # Process final output

    # Check whether the model wants to call a function and call it, if appropriate
    if function_call is not None:

        # Read the function call from model response and execute it (if appropriate)
        available_funs = functions["available_funs"]
        fun_name = function_call.get("name", None)
        if fun_name is not None and fun_name and fun_name in available_funs:
            function = available_funs[fun_name]
        else:
            function = None
        fun_args = function_call.get("arguments", None)
        if fun_args is not None and isinstance(fun_args, str):
            fun_args = json.loads(fun_args)
        if function is not None:
            fun_res = function(fun_args)
        else:
            fun_res = ["Error, no function specified"]

        out_messages = [{"role": "function", "name": fun_name, "content": one_fun_res} for one_fun_res in fun_res]

    else:   # Not a function call, return normal message

        # Sanitize
        if reply_text.startswith("AI: "):
            reply_text = reply_text.split("AI: ", 1)[1]

        out_messages = [{"role": "assistant", "content": reply_text}]