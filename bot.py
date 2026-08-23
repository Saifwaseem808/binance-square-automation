import json
import os
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import requests
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-3.6-flash"

BINANCE_SQUARE_KEY = os.environ["BINANCE_SQUARE_OPENAPI_KEY"]

BASE_DIR = Path.cwd()
CHART_PATH = BASE_DIR / "market_chart.png"


# ============================================================
# BINANCE MARKET DATA
# ============================================================

def get_market_data():

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
    ]

    data = {}

    url = "https://data-api.binance.vision/api/v3/ticker/24hr"

    for symbol in symbols:

        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=30,
        )

        response.raise_for_status()

        item = response.json()

        data[symbol] = {
            "price_usd": float(item["lastPrice"]),
            "change_24h_percent": float(
                item["priceChangePercent"]
            ),
            "high_24h": float(item["highPrice"]),
            "low_24h": float(item["lowPrice"]),
            "volume_24h": float(item["volume"]),
            "quote_volume_24h": float(
                item["quoteVolume"]
            ),
        }

    return data


# ============================================================
# NEWS COLLECTION
# ============================================================

def fetch_news(query, limit=6):

    encoded = urllib.parse.quote_plus(query)

    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}"
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
        timeout=30,
    ) as response:

        xml_data = response.read()

    root = ET.fromstring(xml_data)

    articles = []

    for item in root.findall(".//item")[:limit]:

        title = item.findtext(
            "title",
            default="",
        ).strip()

        link = item.findtext(
            "link",
            default="",
        ).strip()

        description = item.findtext(
            "description",
            default="",
        ).strip()

        published = item.findtext(
            "pubDate",
            default="",
        ).strip()

        source_node = item.find("source")

        source = ""

        if source_node is not None:
            source = (
                source_node.text or ""
            ).strip()

        if title and link:

            articles.append(
                {
                    "title": title,
                    "url": link,
                    "description": description,
                    "published": published,
                    "source": source,
                }
            )

    return articles


def get_latest_news():

    queries = [
        "Bitcoin crypto",
        "Ethereum crypto",
        "Binance BNB crypto",
        "crypto ETF",
        "crypto regulation",
    ]

    unique = {}

    for query in queries:

        try:

            articles = fetch_news(
                query,
                limit=5,
            )

            for article in articles:

                if article["url"] not in unique:

                    unique[
                        article["url"]
                    ] = article

        except Exception as error:

            print(
                f"News query failed: {query}"
            )

            print(error)

    news = list(unique.values())

    print(
        f"Collected {len(news)} news items."
    )

    return news[:20]


# ============================================================
# MARKET CHART
# ============================================================

def create_market_chart(market_data):

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
    ]

    labels = [
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
        labels,
        changes,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "BTC, ETH & BNB — 24H Performance",
        fontsize=18,
        fontweight="bold",
    )

    ax.set_ylabel(
        "24H Change (%)"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        changes,
    ):

        if value >= 0:
            y = value + 0.15
            alignment = "bottom"
        else:
            y = value - 0.15
            alignment = "top"

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            y,
            f"{value:+.2f}%",
            ha="center",
            va=alignment,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.02,
        "Source: Binance market data",
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )

    plt.savefig(
        CHART_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

    if not CHART_PATH.exists():
        raise RuntimeError(
            "Chart was not created."
        )

    return CHART_PATH


# ============================================================
# GEMINI HELPER
# ============================================================

def ask_gemini(prompt):

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned empty output."
        )

    try:
        return json.loads(
            response.text
        )

    except json.JSONDecodeError as error:

        print(response.text)

        raise RuntimeError(
            "Gemini returned invalid JSON."
        ) from error


# ============================================================
# 1. NEWS POST
# ============================================================

def generate_news_post(news):

    news_data = []

    for item in news[:10]:

        news_data.append(
            {
                "title": item["title"],
                "source": item["source"],
                "published": item["published"],
                "description": item["description"],
                "url": item["url"],
            }
        )

    prompt = f"""
Create ONE Binance Square NEWS POST.

This is ONLY a news post.

Do NOT write a market analysis.

Do NOT discuss BTC/ETH/BNB price performance
unless the supplied news itself is specifically
about that price movement.

Do NOT create a long-form article.

Select the 3-5 most important fresh crypto news
items from the supplied list.

For each selected story:

- State what happened.
- Explain why it matters.
- Keep it concise.
- Clearly identify the source.
- Do not invent facts.
- Do not copy article text.
- Do not make unsupported predictions.

Use professional English.

Add relevant $TOKEN tags and 3-5 hashtags.

Return JSON:

{{
  "title": "News headline",
  "body": "News-only post"
}}

NEWS:

{json.dumps(news_data, indent=2)}
"""

    result = ask_gemini(prompt)

    return (
        result["title"].strip(),
        result["body"].strip(),
    )


# ============================================================
# 2. MARKET ANALYSIS POST
# ============================================================

def generate_market_analysis(
    market_data
):

    prompt = f"""
Create ONE Binance Square MARKET ANALYSIS POST.

This is ONLY a market analysis.

Do NOT create a news roundup.

Do NOT include external news.

Analyze ONLY the supplied Binance market data.

Discuss:

- BTC price and 24h movement
- ETH price and 24h movement
- BNB price and 24h movement
- Relative strength
- Volatility
- Volume
- High/low ranges
- Bullish and bearish scenarios
- Key risks
- What market participants should watch

Do NOT guarantee any outcome.

Do NOT promise profits.

Do NOT say buy or sell.

Use $BTC, $ETH and $BNB naturally.

Add 3-5 relevant hashtags.

Target approximately 500-700 words.

Return JSON:

{{
  "title": "Market analysis title",
  "body": "Market-analysis-only post"
}}

MARKET DATA:

{json.dumps(market_data, indent=2)}
"""

    result = ask_gemini(prompt)

    return (
        result["title"].strip(),
        result["body"].strip(),
    )


# ============================================================
# 3. ORIGINAL LONG-FORM ARTICLE
# ============================================================

def generate_original_article(
    market_data,
    news,
):

    news_titles = [
        item["title"]
        for item in news[:10]
    ]

    prompt = f"""
Create ONE ORIGINAL LONG-FORM CRYPTO ARTICLE
for Binance Square.

This must be completely different from:

1. The news post.
2. The market analysis post.

Do NOT create a news roundup.

Do NOT simply describe today's BTC/ETH/BNB prices.

Choose ONE educational crypto topic that is
useful to Binance Square readers.

Examples:

- How liquidity affects crypto markets
- How ETF flows can influence sentiment
- How to understand crypto market cycles
- How leverage creates volatility
- How investors can interpret market volume
- How macroeconomic expectations affect crypto

Choose the topic based on what is most relevant,
but do not invent facts.

The article should:

- Have a strong title.
- Be educational.
- Explain concepts deeply.
- Use examples where appropriate.
- Include risks and limitations.
- Be original.
- Be approximately 1000-1400 words.
- Use $BTC, $ETH and $BNB only when relevant.
- End with 3-5 relevant hashtags.

IMPORTANT:

The supplied news titles are ONLY context for
choosing a relevant topic.

Do NOT turn these headlines into a news article.

Do NOT copy them.

Do NOT mention every headline.

Return JSON:

{{
  "title": "Original article title",
  "body": "Complete long-form article"
}}

CURRENT MARKET DATA:

{json.dumps(market_data, indent=2)}

RECENT NEWS HEADLINES FOR CONTEXT ONLY:

{json.dumps(news_titles, indent=2)}
"""

    result = ask_gemini(prompt)

    return (
        result["title"].strip(),
        result["body"].strip(),
    )


# ============================================================
# BINANCE SQUARE PUBLISHER
# ============================================================

def publish_article(
    title,
    body,
    cover=None,
):

    env = os.environ.copy()

    env[
        "BINANCE_SQUARE_OPENAPI_KEY"
    ] = BINANCE_SQUARE_KEY

    if cover:

        cover_path = Path(
            cover
        ).resolve()

        if not cover_path.exists():

            raise RuntimeError(
                f"Cover not found: {cover_path}"
            )

        command = [
            "node",
            "scripts/post-image.mjs",
            "--text",
            body,
            "--title",
            title,
            "--cover",
            str(cover_path),
        ]

    else:

        command = [
            "node",
            "scripts/post-text.mjs",
            "--text",
            body,
            "--title",
            title,
        ]

    result = subprocess.run(
        command,
        cwd="./square-post",
        env=env,
        text=True,
        capture_output=True,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

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
        " BINANCE SQUARE 3-POST AUTOMATION"
    )
    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("")
    print("1/7 Getting market data...")

    market_data = get_market_data()

    print(
        "Market data ready."
    )

    print("")
    print("2/7 Getting fresh news...")

    news = get_latest_news()

    if not news:

        raise RuntimeError(
            "No news was collected."
        )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------

    print("")
    print("3/7 Creating market chart...")

    chart = create_market_chart(
        market_data
    )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    print("")
    print(
        "4/7 Generating separate NEWS post..."
    )

    news_title, news_body = (
        generate_news_post(news)
    )

    print(
        f"NEWS: {news_title}"
    )

    print(
        "Publishing NEWS post..."
    )

    publish_article(
        news_title,
        news_body,
    )

    print(
        "NEWS POST PUBLISHED."
    )

    # --------------------------------------------------------
    # MARKET ANALYSIS
    # --------------------------------------------------------

    print("")
    print(
        "5/7 Generating separate MARKET ANALYSIS..."
    )

    analysis_title, analysis_body = (
        generate_market_analysis(
            market_data
        )
    )

    print(
        f"MARKET: {analysis_title}"
    )

    print(
        "Publishing MARKET ANALYSIS..."
    )

    publish_article(
        analysis_title,
        analysis_body,
        chart,
    )

    print(
        "MARKET ANALYSIS PUBLISHED."
    )

    # --------------------------------------------------------
    # ARTICLE
    # --------------------------------------------------------

    print("")
    print(
        "6/7 Generating separate LONG-FORM ARTICLE..."
    )

    article_title, article_body = (
        generate_original_article(
            market_data,
            news,
        )
    )

    print(
        f"ARTICLE: {article_title}"
    )

    print(
        "Publishing LONG-FORM ARTICLE..."
    )

    # The existing market chart is NOT used
    # for this article.
    # This keeps the article separate from
    # the market-analysis post.

    publish_article(
        article_title,
        article_body,
    )

    print(
        "LONG-FORM ARTICLE PUBLISHED."
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("")
    print(
        "7/7 All three posts published."
    )

    print("")
    print(
        "=========================================="
    )
    print(
        " SUCCESS — 3 SEPARATE POSTS"
    )
    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
