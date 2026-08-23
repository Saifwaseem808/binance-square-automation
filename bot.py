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
# GEMINI JSON HELPER
# ============================================================

def ask_gemini_json(prompt):

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
            "Gemini returned invalid JSON:"
        )

        print(
            response.text
        )

        raise RuntimeError(
            "Gemini returned invalid JSON."
        ) from error


# ============================================================
# GEMINI PLAIN TEXT HELPER
# ============================================================

def ask_gemini_text(prompt):

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned empty article."
        )

    return response.text.strip()


# ============================================================
# CLEAN BINANCE SQUARE TEXT
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

    # Keep only one occurrence of each
    # supported token cashtag.
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

BINANCE TAG RULES:

Use at most 3 token tags.

Allowed:

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

Add 3 relevant hashtags.

Return JSON only:

{{
    "title": "News title",
    "body": "News post"
}}

NEWS:

{json.dumps(news_data, indent=2)}
"""

    result = ask_gemini_json(
        prompt
    )

    title = clean_square_text(
        result["title"]
    )

    body = clean_square_text(
        result["body"]
    )

    return title, body


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

Do not give guaranteed predictions.

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

    title = clean_square_text(
        result["title"]
    )

    body = clean_square_text(
        result["body"]
    )

    return title, body


# ============================================================
# ORIGINAL LONG-FORM ARTICLE
# ============================================================

def generate_original_article():

    prompt = """
Write ONE ORIGINAL LONG-FORM CRYPTO EDUCATIONAL ARTICLE
for Binance Square.

IMPORTANT:

This is NOT a news post.

This is NOT a daily market analysis.

Do NOT summarize today's headlines.

Choose ONE useful educational topic.

Choose from topics such as:

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

- Original writing.
- Approximately 1000-1400 words.
- Clear sections.
- Useful to intermediate crypto readers.
- Explain concepts with practical examples.
- Include risks and limitations.
- No guaranteed returns.
- No financial guarantees.
- No "buy now" or "sell now".
- Use $BTC, $ETH or $BNB only where genuinely relevant.
- Use at most 3 token tags.
- Add 3 relevant hashtags at the end.

VERY IMPORTANT:

Do NOT return JSON.

Return the article using EXACTLY this format:

TITLE:
Your title here

BODY:
Your complete article here

Do not write anything before TITLE:.

Do not write anything after the article.

Do not use JSON.
"""

    text = ask_gemini_text(
        prompt
    )

    # --------------------------------------------------------
    # Extract TITLE
    # --------------------------------------------------------

    title_match = re.search(
        r"^\s*TITLE:\s*(.+?)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )

    if not title_match:

        raise RuntimeError(
            "Gemini article did not contain TITLE:."
        )

    title = title_match.group(
        1
    ).strip()

    # --------------------------------------------------------
    # Extract BODY
    # --------------------------------------------------------

    body_match = re.search(
        r"\bBODY:\s*(.*)$",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if not body_match:

        raise RuntimeError(
            "Gemini article did not contain BODY:."
        )

    body = body_match.group(
        1
    ).strip()

    if not title:

        raise RuntimeError(
            "Article title is empty."
        )

    if not body:

        raise RuntimeError(
            "Article body is empty."
        )

    title = clean_square_text(
        title
    )

    body = clean_square_text(
        body
    )

    return title, body


# ============================================================
# BINANCE SQUARE PUBLISH
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
        " BINANCE SQUARE 3-POST AUTOMATION"
    )
    print(
        "=========================================="
    )

    # ========================================================
    # MARKET DATA
    # ========================================================

    print("")
    print(
        "1/7 Getting market data..."
    )

    market_data = get_market_data()

    print(
        "Market data ready."
    )

    # ========================================================
    # NEWS
    # ========================================================

    print("")
    print(
        "2/7 Getting fresh news..."
    )

    news = get_latest_news()

    if not news:

        raise RuntimeError(
            "No news was collected."
        )

    # ========================================================
    # CHART
    # ========================================================

    print("")
    print(
        "3/7 Creating market chart..."
    )

    chart = create_market_chart(
        market_data
    )

    # ========================================================
    # NEWS POST
    # ========================================================

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

    # ========================================================
    # MARKET ANALYSIS
    # ========================================================

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

    # ========================================================
    # ORIGINAL ARTICLE
    # ========================================================

    print("")
    print(
        "6/7 Creating ORIGINAL ARTICLE..."
    )

    article_title, article_body = (
        generate_original_article()
    )

    print(
        f"ARTICLE: {article_title}"
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

    # ========================================================
    # DONE
    # ========================================================

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
