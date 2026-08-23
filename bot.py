import json
import os
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from google import genai
from google.genai import types
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-3.6-flash"

BINANCE_SQUARE_KEY = os.environ["BINANCE_SQUARE_OPENAPI_KEY"]

BASE_DIR = Path.cwd()
CHART_PATH = BASE_DIR / "market_chart.png"


# ============================================================
# MARKET DATA
# ============================================================

def get_market_data():

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
    ]

    market_data = {}

    url = "https://data-api.binance.vision/api/v3/ticker/24hr"

    for symbol in symbols:

        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        market_data[symbol] = {
            "price_usd": float(data["lastPrice"]),
            "change_24h_percent": float(data["priceChangePercent"]),
            "high_24h": float(data["highPrice"]),
            "low_24h": float(data["lowPrice"]),
            "volume_24h": float(data["volume"]),
            "quote_volume_24h": float(data["quoteVolume"]),
        }

    return market_data


# ============================================================
# NEWS COLLECTION
# ============================================================

def fetch_news(query, limit=6):

    encoded_query = urllib.parse.quote_plus(query)

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        "&hl=en-US"
        "&gl=US"
        "&ceid=US:en"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        xml_data = response.read()

    root = ET.fromstring(xml_data)

    articles = []

    for item in root.findall(".//item")[:limit]:

        title = item.findtext(
            "title",
            default=""
        ).strip()

        link = item.findtext(
            "link",
            default=""
        ).strip()

        description = item.findtext(
            "description",
            default=""
        ).strip()

        pub_date = item.findtext(
            "pubDate",
            default=""
        ).strip()

        source_element = item.find(
            "source"
        )

        source = ""

        if source_element is not None:
            source = (
                source_element.text or ""
            ).strip()

        if title and link:

            articles.append(
                {
                    "title": title,
                    "source": source,
                    "published": pub_date,
                    "description": description,
                    "url": link,
                }
            )

    return articles


def get_latest_news():

    queries = [
        "Bitcoin crypto market",
        "Ethereum crypto market",
        "BNB Binance crypto",
        "crypto ETF regulation",
        "crypto market macro liquidity",
    ]

    all_news = []

    for query in queries:

        try:

            news = fetch_news(
                query,
                limit=5
            )

            all_news.extend(news)

        except Exception as error:

            print(
                f"News query failed: {query}"
            )

            print(error)

    # Remove duplicate URLs
    unique_news = {}

    for article in all_news:

        url = article["url"]

        if url not in unique_news:

            unique_news[url] = article

    news = list(
        unique_news.values()
    )

    # Keep the newest articles when publication
    # timestamps are available.
    news = news[:25]

    print("")
    print(
        f"Collected {len(news)} news items."
    )

    for article in news[:10]:

        print(
            f"- {article['source']}: "
            f"{article['title']}"
        )

    return news


# ============================================================
# CREATE MARKET CHART
# ============================================================

def create_market_chart(market_data):

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
    ]

    names = [
        "BTC",
        "ETH",
        "BNB",
    ]

    changes = [
        market_data[symbol][
            "change_24h_percent"
        ]
        for symbol in symbols
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars = ax.bar(
        names,
        changes
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_title(
        "BTC, ETH & BNB — 24H Market Performance",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_ylabel(
        "24H Change (%)"
    )

    ax.set_xlabel(
        "Asset"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        changes
    ):

        if value >= 0:

            position = (
                value + 0.15
            )

            alignment = "bottom"

        else:

            position = (
                value - 0.15
            )

            alignment = "top"

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            position,
            f"{value:+.2f}%",
            ha="center",
            va=alignment,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.02,
        "Source: Binance market data • "
        "Snapshot at publication time",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    plt.savefig(
        CHART_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    if not CHART_PATH.exists():

        raise RuntimeError(
            "Chart file was not created."
        )

    print(
        f"Chart created: {CHART_PATH}"
    )

    return CHART_PATH


# ============================================================
# GEMINI NEWS + MARKET ANALYSIS
# ============================================================

def generate_article(
    market_data,
    news,
):

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    news_for_ai = []

    for article in news:

        news_for_ai.append(
            {
                "title": article["title"],
                "source": article["source"],
                "published": article["published"],
                "description": article["description"],
                "url": article["url"],
            }
        )

    prompt = f"""
You are a professional cryptocurrency market analyst
writing an ORIGINAL daily article for Binance Square.

Your job is to combine:

1. Current Binance market data.
2. Fresh crypto news supplied below.

Do NOT copy news articles.

Do NOT invent facts.

Do NOT treat an unverified headline as confirmed fact.

If sources disagree, mention uncertainty.

Only use information supplied in the market data
and news list.

==================================================
MARKET DATA
==================================================

{json.dumps(market_data, indent=2)}

==================================================
LATEST NEWS
==================================================

{json.dumps(news_for_ai, indent=2)}

==================================================
ARTICLE REQUIREMENTS
==================================================

Write a useful market-analysis article.

Include:

1. Strong factual title.
2. Executive market overview.
3. Bitcoin analysis.
4. Ethereum analysis.
5. BNB analysis.
6. Important news and why it matters.
7. Macro/liquidity/regulatory context when supported.
8. Market sentiment.
9. Important risks.
10. What traders/investors should watch next.
11. A concise conclusion.

Do not provide guaranteed predictions.

Do not promise profits.

Do not say "buy now", "100% profit",
"guaranteed", or similar promotional claims.

Clearly distinguish:

- confirmed information
- market interpretation
- potential scenarios

Use professional English.

Target approximately 900-1300 words.

==================================================
BINANCE TAGS
==================================================

Naturally use:

$BTC

$ETH

$BNB

Only use a token tag if that asset is actually discussed.

At the END add 3-5 relevant hashtags.

Examples:

#Bitcoin
#Ethereum
#BNB
#Crypto
#MarketAnalysis

Do not invent unrelated hashtags.

==================================================
SOURCE REFERENCES
==================================================

At the end include a small section:

Sources:
- Source Name — headline

Use only sources supplied above.

Do NOT invent URLs.

==================================================
OUTPUT
==================================================

Return ONLY valid JSON:

{{
    "title": "Article title",
    "body": "Complete article body"
}}

Current UTC time:

{datetime.now(timezone.utc).isoformat()}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    text = response.text

    if not text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:

        article = json.loads(text)

    except json.JSONDecodeError as error:

        print(
            "Gemini returned invalid JSON:"
        )

        print(text)

        raise RuntimeError(
            "Gemini response was not valid JSON."
        ) from error

    if "title" not in article:

        raise RuntimeError(
            "Gemini response has no title."
        )

    if "body" not in article:

        raise RuntimeError(
            "Gemini response has no body."
        )

    title = str(
        article["title"]
    ).strip()

    body = str(
        article["body"]
    ).strip()

    if not title:

        raise RuntimeError(
            "Generated title is empty."
        )

    if not body:

        raise RuntimeError(
            "Generated article body is empty."
        )

    return title, body


# ============================================================
# VALIDATE TAGS
# ============================================================

def validate_tags(body):

    upper_body = body.upper()

    detected_tokens = []

    for token in [
        "$BTC",
        "$ETH",
        "$BNB",
    ]:

        if token in upper_body:

            detected_tokens.append(
                token
            )

    detected_hashtags = []

    for word in body.split():

        clean = word.strip(
            ".,!?;:()[]{}\"'"
        )

        if clean.startswith("#"):

            detected_hashtags.append(
                clean
            )

    detected_hashtags = list(
        dict.fromkeys(
            detected_hashtags
        )
    )

    print("")
    print(
        "Detected token tags:"
    )

    print(
        ", ".join(
            detected_tokens
        )
        if detected_tokens
        else "None"
    )

    print("")
    print(
        "Detected hashtags:"
    )

    print(
        ", ".join(
            detected_hashtags
        )
        if detected_hashtags
        else "None"
    )

    return body


# ============================================================
# PUBLISH TO BINANCE SQUARE
# ============================================================

def publish_to_square(
    title,
    body,
    cover_path,
):

    env = os.environ.copy()

    env[
        "BINANCE_SQUARE_OPENAPI_KEY"
    ] = BINANCE_SQUARE_KEY

    absolute_cover_path = Path(
        cover_path
    ).resolve()

    if not absolute_cover_path.exists():

        raise RuntimeError(
            f"Cover image does not exist: "
            f"{absolute_cover_path}"
        )

    print("")
    print(
        f"Cover image found: "
        f"{absolute_cover_path}"
    )

    command = [
        "node",
        "scripts/post-image.mjs",
        "--text",
        body,
        "--title",
        title,
        "--cover",
        str(absolute_cover_path),
    ]

    print("")
    print(
        "Publishing article..."
    )

    result = subprocess.run(
        command,
        cwd="./square-post",
        env=env,
        text=True,
        capture_output=True,
    )

    print("")
    print(
        "Binance Square output:"
    )

    print(
        result.stdout
    )

    if result.stderr:

        print("")
        print(
            "Binance Square warnings:"
        )

        print(
            result.stderr
        )

    if result.returncode != 0:

        raise RuntimeError(
            "Binance Square publishing failed."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print(
        "=========================================="
    )

    print(
        "    BINANCE SQUARE NEWS AUTOMATION"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # 1. MARKET DATA
    # --------------------------------------------------------

    print("")
    print(
        "1/6 Getting Binance market data..."
    )

    market_data = get_market_data()

    print(
        "Market data received."
    )

    # --------------------------------------------------------
    # 2. NEWS
    # --------------------------------------------------------

    print("")
    print(
        "2/6 Collecting latest crypto news..."
    )

    news = get_latest_news()

    if not news:

        raise RuntimeError(
            "No news was collected."
        )

    # --------------------------------------------------------
    # 3. CHART
    # --------------------------------------------------------

    print("")
    print(
        "3/6 Creating market chart..."
    )

    chart_path = create_market_chart(
        market_data
    )

    # --------------------------------------------------------
    # 4. AI ANALYSIS
    # --------------------------------------------------------

    print("")
    print(
        "4/6 Generating news + market analysis..."
    )

    title, body = generate_article(
        market_data,
        news,
    )

    print("")
    print(
        "Generated title:"
    )

    print(title)

    # --------------------------------------------------------
    # 5. TAGS
    # --------------------------------------------------------

    print("")
    print(
        "5/6 Checking token tags and hashtags..."
    )

    body = validate_tags(
        body
    )

    # --------------------------------------------------------
    # 6. PUBLISH
    # --------------------------------------------------------

    print("")
    print(
        "6/6 Publishing to Binance Square..."
    )

    publish_to_square(
        title,
        body,
        chart_path,
    )

    print("")
    print(
        "=========================================="
    )

    print(
        "       PUBLISHED SUCCESSFULLY"
    )

    print(
        "=========================================="
    )

    print("")


if __name__ == "__main__":

    main()
