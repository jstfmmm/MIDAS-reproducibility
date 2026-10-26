import os
from typing import List, Optional

import requests

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY", "").strip()

SERPER_URL = "https://google.serper.dev/search"
DEFAULT_SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()


def filter_urls(urls):
    urls_new = []
    for url in urls:
        if url in urls_new or f"http://{url}" in urls_new or f"https://{url}" in urls_new:
            continue
        if not url.startswith("http"):
            url = f"http://{url}"
        urls_new.append(url)
    return urls_new


def serp_api(query: str, language: str):
    if not SERPAPI_KEY:
        raise RuntimeError("SERPAPI_API_KEY is required for the SerpAPI retriever")
    params = {"engine": "bing", "q": query, "api_key": SERPAPI_KEY}
    response = requests.get("https://serpapi.com/search", params=params, timeout=30)
    if response.status_code != 200:
        raise Exception("serpapi returned %d\n%s" % (response.status_code, response.text))
    result = response.json()
    urls, info = [], []
    for item in result.get("organic_results", []):
        if "title" not in item or "link" not in item or "snippet" not in item:
            continue
        info.append({"title": item['title'], "url": item['link'], "snip": item['snippet']})
        urls.append(item["link"])

    return urls, info


def _serper_request_headers(api_key: Optional[str] = None):
    key = (api_key or DEFAULT_SERPER_API_KEY).strip()
    if not key:
        raise RuntimeError("SERPER_API_KEY is required for the web retriever")
    return {"X-API-KEY": key, "Content-Type": "application/json"}


def serper_api(query: str, language: str, api_key: Optional[str] = None):
    payload = {
        "q": query,
        "gl": "cn" if language == "zh" else "us",
        "hl": "zh-cn" if language == "zh" else "en",
    }
    response = requests.post(
        SERPER_URL,
        headers=_serper_request_headers(api_key),
        json=payload,
        verify=False,
        timeout=12,
    )

    if response.status_code != 200:
        raise Exception("serper api returned %d\n%s" % (response.status_code, response.text))

    result = response.json()
    urls = []
    info = {"organic": [], "peopleAlsoAsk": result.get("peopleAlsoAsk", []),
            "relatedSearches": result.get("relatedSearches", [])}

    for item in result.get("organic", []):
        if "title" not in item or "link" not in item or "snippet" not in item:
            continue
        urls.append(item["link"])
        info["organic"].append(item)

    return urls, info


def serper_batch_api(queries: List[str], language_list: List[str]):
    payloads = [
        {"q": query, "gl": "cn" if language == "zh" else "us",
         "hl": "zh-cn" if language == "zh" else "en"}
        for query, language in zip(queries, language_list)
    ]
    response = requests.post(
        SERPER_URL,
        headers=_serper_request_headers(),
        json=payloads,
        timeout=30,
    )

    if response.status_code != 200:
        raise Exception("serper api returned %d\n%s" % (response.status_code, response.text))

    results = response.json()
    urls_list, infos = [], []

    for result in results:
        info = {"organic": [], "peopleAlsoAsk": result.get("peopleAlsoAsk", []),
                "relatedSearches": result.get("relatedSearches", [])}
        urls = []

        for item in result.get("organic", []):
            if "title" not in item or "link" not in item or "snippet" not in item:
                continue
            urls.append(item["link"])
            info["organic"].append(item)
        urls_list.append(urls)
        infos.append(info)

    return urls_list, infos
