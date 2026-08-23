import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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

BINANCE_SQUARE_KEY = os.environ[
    "BINANCE_SQUARE_OPENAPI_KEY"
]

POST_TYPE = os.environ.get(
    "POST_TYPE",
    "ALL"
).upper()

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

    url = (
        "https://data-api.binance.vision"
        "/api/v3/ticker/24hr"
    )

    for symbol in symbols:

        response = requests.get(
            url,
            params={"symbol": symbol},
            timeout=30,
        )

        response.raise_for_status()

        item = response.json()

        market_data[symbol] = {
            "price_usd": float(
                item["lastPrice"]
            ),
            "change_24h_percent": float(
                item["priceChangePercent"]
            ),
            "high_24h": float(
                item["highPrice"]
            ),
            "low_24h": float(
                item["lowPrice"]
            ),
            "volume_24h": float(
                item["volume"]
            ),
            "quote_volume_24h": float(
                item["quoteVolume"]
            ),
        }

    return market_data


# ============================================================
# NEWS COLLECTION
# ============================================================

def fetch_news(query, limit=5):

    encoded_query = urllib.parse.quote_plus(
        query
    )

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

                url = article["url"]

                if url not in unique:

                    unique[url] = article

        except Exception as error:

            print(
                f"News query failed: {query}"
            )

            print(error)

    news = list(
        unique.values()
    )

    print(
        f"Collected {len(news)} news items."
    )

    return news[:20]


# ============================================================
# MARKET CHART
# ============================================================

def create_market_chart(
    market_data
):

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
# GEMINI JSON
# ============================================================

def ask_gemini_text(prompt):

    import time

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    # Primary model + fallback model
    models = [
        GEMINI_MODEL,
        "gemini-2.5-flash",
    ]

    last_error = None

    for model in models:

        for attempt in range(3):

            try:

                print(
                    f"Gemini model: {model} "
                    f"| attempt {attempt + 1}/3"
                )

                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                if response.text:

                    print(
                        f"Gemini generation successful "
                        f"using {model}"
                    )

                    return response.text.strip()

                raise RuntimeError(
                    "Gemini returned empty response."
                )

            except Exception as error:

                last_error = error

                error_text = str(error)

                print("")
                print(
                    f"Gemini attempt failed: "
                    f"{error_text}"
                )

                # Retry temporary server/rate-limit errors.
                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    if attempt < 2:

                        wait_seconds = (
                            5 * (attempt + 1)
                        )

                        print(
                            f"Temporary Gemini error. "
                            f"Waiting {wait_seconds}s..."
                        )

                        time.sleep(
                            wait_seconds
                        )

                        continue

                    print(
                        f"Model {model} failed "
                        f"after 3 attempts."
                    )

                    break

                # Other errors should not be
                # hidden/retried unnecessarily.
                raise

    raise RuntimeError(
        "Gemini article generation failed "
        "after retries and fallback models. "
        f"Last error: {last_error}"
    )


# ============================================================
# BINANCE TEXT CLEANER
# ============================================================

def clean_square_text(text):

    cleaned = str(text)

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

    pattern = re.compile(
        r"\$(BTC|ETH|BNB)\b",
        re.IGNORECASE,
    )

    seen = set()

    def replace_token(match):

        token = match.group(1).upper()

        if token in seen:
            return token

        seen.add(token)

        return f"${token}"

    cleaned = pattern.sub(
        replace_token,
        cleaned,
    )

    return cleaned.strip()


# ============================================================
# NEWS
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
Create ONE SHORT Binance Square NEWS POST.

This is NEWS ONLY.

Do NOT create market analysis.

Do NOT create an educational article.

Select the most important 3-5 fresh stories.

For each:

- What happened
- Why it matters
- Source

Do not invent facts.

Do not copy article text.

Use at most 3 token tags.

Allowed:

$BTC
$ETH
$BNB

Never use trading pairs such as BTCUSDT,
ETHUSDT, BNBUSDT, BTC/USDT or ETH/USDT.

Add 3 relevant hashtags.

Return JSON only:

{{
    "title": "News title",
    "body": "News post"
}}

NEWS:

{json.dumps(
    news_data,
    indent=2
)}
"""

    result = ask_gemini_json(
        prompt
    )

    return (
        clean_square_text(
            result["title"]
        ),
        clean_square_text(
            result["body"]
        ),
    )


# ============================================================
# MARKET ANALYSIS
# ============================================================

def generate_market_analysis(
    market_data
):

    prompt = f"""
Create ONE Binance Square MARKET ANALYSIS POST.

This is ONLY market analysis.

Do NOT include external news.

Use ONLY this Binance data:

{json.dumps(
    market_data,
    indent=2
)}

Analyze:

- BTC price
- BTC 24h change
- ETH price
- ETH 24h change
- BNB price
- BNB 24h change
- Volume
- High/low ranges
- Relative strength
- Volatility
- Bullish scenario
- Bearish scenario
- Risks
- What to watch

Do not promise profits.

Do not guarantee predictions.

Do not tell users to buy or sell.

Use $BTC, $ETH and $BNB at most once each.

Do not use trading pairs.

Add 3 relevant hashtags.

Return JSON only:

{{
    "title": "Market analysis title",
    "body": "Market analysis post"
}}
"""

    result = ask_gemini_json(
        prompt
    )

    return (
        clean_square_text(
            result["title"]
        ),
        clean_square_text(
            result["body"]
        ),
    )


# ============================================================
# ORIGINAL ARTICLE
# ============================================================

def generate_original_article():

    prompt = """
Write ONE ORIGINAL LONG-FORM CRYPTO
EDUCATIONAL ARTICLE for Binance Square.

This is NOT a news post.

This is NOT a daily market analysis.

Choose ONE useful educational topic.

Possible topics:

- Crypto liquidity
- Market cycles
- Risk management
- Leverage
- ETF mechanics
- Crypto volatility
- Market depth
- Trading psychology
- Macro factors
- Understanding crypto volume

Choose ONE topic only.

Requirements:

- Original writing
- Approximately 1000-1400 words
- Clear sections
- Useful to intermediate crypto readers
- Practical examples
- Risks and limitations
- No guaranteed returns
- No "buy now" or "sell now"
- At most 3 token tags
- 3 relevant hashtags

Do NOT return JSON.

Return EXACTLY:

TITLE:
Your title

BODY:
Your complete article

Nothing before TITLE.
"""

    text = ask_gemini_text(
        prompt
    )

    title_match = re.search(
        r"^\s*TITLE:\s*(.+?)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )

    if not title_match:

        raise RuntimeError(
            "Article did not contain TITLE:."
        )

    title = title_match.group(
        1
    ).strip()

    body_match = re.search(
        r"\bBODY:\s*(.*)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if not body_match:

        raise RuntimeError(
            "Article did not contain BODY:."
        )

    body = body_match.group(
        1
    ).strip()

    if not title or not body:

        raise RuntimeError(
            "Article title or body is empty."
        )

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
                f"Cover not found: "
                f"{cover_path}"
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
        " BINANCE SQUARE SCHEDULED AUTOMATION"
    )
    print(
        f" POST TYPE: {POST_TYPE}"
    )
    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    if POST_TYPE == "NEWS":

        print("")
        print(
            "Getting fresh news..."
        )

        news = get_latest_news()

        if not news:

            raise RuntimeError(
                "No news was collected."
            )

        title, body = (
            generate_news_post(
                news
            )
        )

        publish_to_square(
            title,
            body,
        )

        print(
            "NEWS POST SUCCESS."
        )

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    elif POST_TYPE == "MARKET":

        print("")
        print(
            "Getting market data..."
        )

        market_data = (
            get_market_data()
        )

        print(
            "Creating market chart..."
        )

        chart = create_market_chart(
            market_data
        )

        print(
            "Generating market analysis..."
        )

        title, body = (
            generate_market_analysis(
                market_data
            )
        )

        publish_to_square(
            title,
            body,
            chart,
        )

        print(
            "MARKET ANALYSIS SUCCESS."
        )

    # --------------------------------------------------------
    # ARTICLE
    # --------------------------------------------------------

    elif POST_TYPE == "ARTICLE":

        print("")
        print(
            "Generating original article..."
        )

        title, body = (
            generate_original_article()
        )

        publish_to_square(
            title,
            body,
        )

        print(
            "ORIGINAL ARTICLE SUCCESS."
        )

    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    elif POST_TYPE == "ALL":

        print(
            "Running ALL mode..."
        )

        market_data = (
            get_market_data()
        )

        news = get_latest_news()

        chart = create_market_chart(
            market_data
        )

        news_title, news_body = (
            generate_news_post(
                news
            )
        )

        publish_to_square(
            news_title,
            news_body,
        )

        analysis_title, analysis_body = (
            generate_market_analysis(
                market_data
            )
        )

        publish_to_square(
            analysis_title,
            analysis_body,
            chart,
        )

        article_title, article_body = (
            generate_original_article()
        )

        publish_to_square(
            article_title,
            article_body,
        )

        print(
            "ALL POSTS SUCCESS."
        )

    else:

        raise RuntimeError(
            "Invalid POST_TYPE: "
            f"{POST_TYPE}"
        )


if __name__ == "__main__":
    main()
