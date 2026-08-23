import json
import os
import re
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

    market_data = {}

    url = "https://data-api.binance.vision/api/v3/ticker/24hr"

    for symbol in symbols:

        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=30,
        )

        response.raise_for_status()

        item = response.json()

        market_data[symbol] = {
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

    return market_data


# ============================================================
# NEWS
# ============================================================

def fetch_news(query, limit=5):

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
        "BNB Binance crypto",
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

    print(
        f"Chart created: {CHART_PATH}"
    )

    return CHART_PATH


# ============================================================
# GEMINI
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

        print(
            "Gemini returned:"
        )

        print(
            response.text
        )

        raise RuntimeError(
            "Gemini returned invalid JSON."
        ) from error


# ============================================================
# CLEAN TEXT FOR BINANCE SQUARE
# ============================================================

def clean_square_text(text):

    cleaned = str(text)

    # Remove trading-pair formats that Binance
    # may interpret as coin-pair tags.
    replacements = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "BNBUSDT": "BNB",
        "BTC/USDT": "BTC",
        "ETH/USDT": "ETH",
        "BNB/USDT": "BNB",
        "BTC-USDT": "BTC",
        "ETH-USDT": "ETH",
        "BNB-USDT": "BNB",
    }

    for old, new in replacements.items():

        cleaned = cleaned.replace(
            old,
            new,
        )

    # Remove accidental repeated token tags.
    token_pattern = re.compile(
        r"\$(BTC|ETH|BNB)\b",
        re.IGNORECASE,
    )

    seen_tokens = set()

    def token_replacer(match):

        token = match.group(1).upper()

        if token in seen_tokens:

            return token

        seen_tokens.add(token)

        return f"${token}"

    cleaned = token_pattern.sub(
        token_replacer,
        cleaned,
    )

    return cleaned.strip()


# ============================================================
# NEWS POST
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
            }
        )

    prompt = f"""
Create ONE short Binance Square NEWS POST.

This post must contain ONLY fresh crypto news.

DO NOT create a market analysis.

DO NOT create a long-form educational article.

Select the most important 3-5 stories.

For each story:

- Explain what happened.
- Explain why it matters.
- Identify the source.
- Do not invent facts.
- Do not copy article text.
- Keep it concise.

IMPORTANT BINANCE RULES:

Use AT MOST 3 token tags in the entire post.

The ONLY allowed token tags are:

$BTC
$ETH
$BNB

Do not use trading pairs.

Never write:

BTCUSDT
ETHUSDT
BNBUSDT
BTC/USDT
ETH/USDT
BNB/USDT

Do not copy pair names from headlines.

Add only 3 relevant hashtags.

Return ONLY JSON:

{{
    "title": "News title",
    "body": "News-only post"
}}

NEWS:

{json.dumps(news_data, indent=2)}
"""

    result = ask_gemini(prompt)

    title = str(
        result["title"]
    ).strip()

    body = str(
        result["body"]
    ).strip()

    return (
        clean_square_text(title),
        clean_square_text(body),
    )


# ============================================================
# MARKET ANALYSIS
# ============================================================

def generate_market_analysis(
    market_data
):

    prompt = f"""
Create ONE Binance Square MARKET ANALYSIS POST.

This post must contain ONLY market analysis.

Do NOT include news.

Do NOT summarize external headlines.

Use ONLY this Binance market data:

{json.dumps(market_data, indent=2)}

Discuss:

- BTC price
- BTC 24h movement
- ETH price
- ETH 24h movement
- BNB price
- BNB 24h movement
- Relative strength
- Volume
- High/low ranges
- Volatility
- Bullish scenario
- Bearish scenario
- Risks
- What to watch next

Do not promise profits.

Do not guarantee predictions.

Do not tell readers to buy or sell.

Use only these token tags:

$BTC
$ETH
$BNB

Use each at most once.

Do not write trading pairs.

Add 3 relevant hashtags.

Return ONLY JSON:

{{
    "title": "Market analysis title",
    "body": "Market-analysis-only post"
}}
"""

    result = ask_gemini(prompt)

    title = str(
        result["title"]
    ).strip()

    body = str(
        result["body"]
    ).strip()

    return (
        clean_square_text(title),
        clean_square_text(body),
    )


# ============================================================
# ORIGINAL ARTICLE
# ============================================================

def generate_original_article(
    market_data,
    news,
):

    prompt = f"""
Create ONE ORIGINAL LONG-FORM CRYPTO ARTICLE
for Binance Square.

This is NOT a news post.

This is NOT a daily market analysis.

Choose ONE educational topic related to crypto.

Possible topics:

- Crypto liquidity
- Market cycles
- ETF flows
- Crypto volatility
- Leverage
- Risk management
- Macro factors
- On-chain fundamentals
- Understanding crypto volume

Choose ONE topic only.

The article must:

- Be original.
- Be educational.
- Explain the topic deeply.
- Be approximately 1000-1400 words.
- Not copy news.
- Not summarize today's headlines.
- Not become a BTC/ETH/BNB daily price report.
- Mention risks and limitations.
- Use $BTC, $ETH or $BNB only when genuinely relevant.
- Add 3 relevant hashtags.

The news supplied below is ONLY context for
choosing a useful topic.

Do NOT turn the news into the article.

Do NOT copy the headlines.

Return ONLY JSON:

{{
    "title": "Original article title",
    "body": "Complete long-form article"
}}

MARKET DATA:

{json.dumps(market_data, indent=2)}

NEWS CONTEXT:

{json.dumps(
    [item["title"] for item in news[:8]],
    indent=2
)}
"""

    result = ask_gemini(prompt)

    title = str(
        result["title"]
    ).strip()

    body = str(
        result["body"]
    ).strip()

    return (
        clean_square_text(title),
        clean_square_text(body),
    )


# ============================================================
# PUBLISH
# ============================================================

def publish_to_square(
    title,
    body,
    cover=None,
):

    env = os.environ.copy()

    env[
        "BINANCE_SQUARE_OPENAPI_KEY"
    ] = BINANCE_SQUARE_KEY

    title = clean_square_text(
        title
    )

    body = clean_square_text(
        body
    )

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

    print("")
    print(
        f"Publishing: {title}"
    )

    result = subprocess.run(
        command,
        cwd="./square-post",
        env=env,
        text=True,
        capture_output=True,
    )

    print(
        result.stdout
    )

    if result.stderr:

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
        " BINANCE SQUARE 3-POST AUTOMATION"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # 1. DATA
    # --------------------------------------------------------

    print("")
    print(
        "1/7 Getting market data..."
    )

    market_data = get_market_data()

    print(
        "Market data ready."
    )

    # --------------------------------------------------------
    # 2. NEWS
    # --------------------------------------------------------

    print("")
    print(
        "2/7 Getting fresh news..."
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
        "3/7 Creating market chart..."
    )

    chart = create_market_chart(
        market_data
    )

    # --------------------------------------------------------
    # 4. NEWS POST
    # --------------------------------------------------------

    print("")
    print(
        "4/7 Creating NEWS post..."
    )

    news_title, news_body = (
        generate_news_post(
            news
        )
    )

    print(
        "Publishing NEWS..."
    )

    publish_to_square(
        news_title,
        news_body,
    )

    print(
        "NEWS POST SUCCESS."
    )

    # --------------------------------------------------------
    # 5. MARKET ANALYSIS
    # --------------------------------------------------------

    print("")
    print(
        "5/7 Creating MARKET ANALYSIS..."
    )

    analysis_title, analysis_body = (
        generate_market_analysis(
            market_data
        )
    )

    print(
        "Publishing MARKET ANALYSIS..."
    )

    publish_to_square(
        analysis_title,
        analysis_body,
        chart,
    )

    print(
        "MARKET ANALYSIS SUCCESS."
    )

    # --------------------------------------------------------
    # 6. ARTICLE
    # --------------------------------------------------------

    print("")
    print(
        "6/7 Creating ORIGINAL ARTICLE..."
    )

    article_title, article_body = (
        generate_original_article(
            market_data,
            news,
        )
    )

    print(
        "Publishing ORIGINAL ARTICLE..."
    )

    publish_to_square(
        article_title,
        article_body,
    )

    print(
        "ORIGINAL ARTICLE SUCCESS."
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("")
    print(
        "7/7 Finished."
    )

    print("")
    print(
        "=========================================="
    )

    print(
        "  3 SEPARATE POSTS PUBLISHED SUCCESSFULLY"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":

    main()
