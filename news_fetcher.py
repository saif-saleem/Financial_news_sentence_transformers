import requests
from bs4 import BeautifulSoup

def fetch_news_links_yahoo(company_name, max_articles=5):
    query = company_name.replace(" ", "+")
    url = f"https://www.bing.com/news/search?q={query}+site:finance.yahoo.com&FORM=HDRSC6"

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch news links: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)

        if href.startswith("https://finance.yahoo.com") and href not in links:
            if len(text) >= 15:  # basic content filter
                links.append(href)
            if len(links) >= max_articles:
                break

    return links
