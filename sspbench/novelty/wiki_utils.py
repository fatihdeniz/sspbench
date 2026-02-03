"""
Wikipedia utilities for the Novelty Engine.
Functions for interacting with Wikipedia API and processing content.
"""

import requests
from bs4 import BeautifulSoup
from .config import WIKIPEDIA_ACCESS_TOKEN, WIKIPEDIA_CLIENT_ID


def get_pageviews(page_title, start_date="2020040100", end_date="2026010100"):
    """
    Get page view statistics for a Wikipedia page.

    Args:
        page_title: Title of the Wikipedia page
        start_date: Start date in YYYYMMDDHH format
        end_date: End date in YYYYMMDDHH format

    Returns:
        int: Number of page views
    """
    headers = {
        'Authorization': f'Bearer {WIKIPEDIA_ACCESS_TOKEN}',
        'User-Agent': 'wikipagerank',
    }

    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page_title}/daily/{start_date}/{end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        views = sum(item['views'] for item in data['items'])

        print(f"Page view count of {page_title}: {views}")

        return views
    else:
        print(f"Failed to retrieve pageviews data for {page_title}. Status code: {response.status_code}")
        return 0


def clean_str(p):
    """Clean and decode text strings."""
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception:
        return ''


def filter_paragraph(paragraph_lst):
    """Filter paragraphs to ensure they have meaningful content."""
    return [p for p in paragraph_lst if len(p.split(" ")) > 2 and len(p.split(".")) > 1]


def get_page_obs(page):
    """Extract observations (paragraphs) from a page."""
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def search_step(entity, output_more=False):
    """
    Search for a Wikipedia entity and extract content.

    Args:
        entity: Entity name to search for
        output_more: Whether to return more detailed content

    Returns:
        tuple: (observations, entity_name, wiki_url)
    """
    headers = {
        "User-Agent": "autobencher-data-curation/1.0 (research)"
    }

    entity_ = entity.replace(" ", "+")
    search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
    response_text = requests.get(search_url, headers=headers).text
    soup = BeautifulSoup(response_text, features="html.parser")
    result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})

    if result_divs:  # mismatch - found similar entities
        result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
        print(f"Could not find {entity}. Searching for similar entities, {result_titles[0]}, ...")
        # Recursively search for the first similar entity
        obs, entity, wiki_url = search_step(result_titles[0], output_more=output_more)
    else:
        print('Found entity', entity)
        canonical = soup.find("link", rel="canonical")
        wiki_url = canonical["href"] if canonical else response.url

        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):
            # Disambiguation page - try with brackets
            obs, entity, wiki_url = search_step("[" + entity + "]", output_more=output_more)
        else:
            page_ = ""
            for p in page:
                if len(p.split(" ")) > 2:
                    page_ += clean_str(p)
                    if not p.endswith("\n"):
                        page_ += "\n"
            obs = get_page_obs(page_)
            if output_more:
                obs = filter_paragraph(obs)
            else:
                obs = filter_paragraph(obs[:10])

    return obs, entity, wiki_url


def search_related_pages(search_query):
    """
    Search for related Wikipedia pages.

    Args:
        search_query: Query to search for

    Returns:
        list: List of page titles
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_query,
        "srlimit": "max",
        "format": "json"
    }
    headers = {
        "User-Agent": "autobencher-data-curation/1.0 (research)"
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        print("Wikipedia API HTTP error:", response.status_code)
        print(response.text)
        return []

    data = response.json()
    if "query" not in data or "search" not in data["query"]:
        print("Wikipedia API returned unexpected payload:")
        print(data)
        return []

    return [item["title"] for item in data["query"]["search"]]