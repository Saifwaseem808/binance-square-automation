import json
import os
import subprocess
from datetime import datetime, timezone
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
# CREATE MARKET CHART
# ============================================================

def create_market_chart(market_data):

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    names = ["BTC", "ETH", "BNB"]

    changes = [
        market_data[symbol]["change_24h_percent"]
        for symbol in symbols
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.bar(names, changes)

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_title(
        "BTC, ETH & BNB — 24H Market Performance",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_ylabel("24H Change (%)")
    ax.set_xlabel("Asset")

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(bars, changes):

        if value >= 0:
            position = value + 0.15
            alignment = "bottom"
        else:
            position = value - 0.15
            alignment = "top"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            position,
            f"{value:+.2f}%",
            ha="center",
            va=alignment,
            fontweight="bold"
        )

    fig.text(
        0.5,
        0.02,
        "Source: Binance market data • Snapshot at publication time",
        ha="center",
        fontsize=9
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 1]
    )

    plt.savefig(
        CHART_PATH,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    if not CHART_PATH.exists():
        raise RuntimeError(
            "Chart file was not created."
        )

    print(
        f"Chart created successfully: {CHART_PATH}"
    )

    return CHART_PATH


# ============================================================
# GEMINI ARTICLE
# ============================================================

def generate_article(market_data):

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    prompt = f"""
You are a professional cryptocurrency market analyst
writing original content for Binance Square.

Create ONE high-quality daily crypto market article.

Use ONLY the market data supplied below.

Rules:

- Do not invent news.
- Do not invent statistics.
- Do not invent events.
- Do not promise profits.
- Do not give guaranteed predictions.
- Explain that prices are snapshots.
- Analyze BTC, ETH and BNB.
- Explain their 24-hour movements.
- Discuss market strength and weakness.
- Discuss risks.
- Use $BTC, $ETH and $BNB naturally.
- Add 3-5 relevant hashtags.
- Use professional English.
- Avoid excessive emojis.
- Create an interesting factual title.
- Target approximately 700-1000 words.

Return ONLY valid JSON:

{{
    "title": "Article title",
    "body": "Complete article body"
}}

MARKET DATA:

{json.dumps(market_data, indent=2)}

CURRENT UTC TIME:

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

    article = json.loads(text)

    title = str(article["title"]).strip()
    body = str(article["body"]).strip()

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
# PUBLISH TO BINANCE SQUARE
# ============================================================

def publish_to_square(title, body, cover_path):

    env = os.environ.copy()

    env["BINANCE_SQUARE_OPENAPI_KEY"] = BINANCE_SQUARE_KEY

    # Convert the chart path to an absolute path.
    absolute_cover_path = Path(cover_path).resolve()

    if not absolute_cover_path.exists():
        raise RuntimeError(
            f"Cover image does not exist: {absolute_cover_path}"
        )

    print(
        f"Cover image found: {absolute_cover_path}"
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

    print(
        "Publishing article with cover image..."
    )

    result = subprocess.run(
        command,
        cwd="./square-post",
        env=env,
        text=True,
        capture_output=True,
    )

    print("Binance Square output:")
    print(result.stdout)

    if result.stderr:
        print(
            "Binance Square warnings:"
        )
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
    print("==========================================")
    print(" BINANCE SQUARE AUTOMATION")
    print("==========================================")

    print("")
    print("1/4 Getting Binance market data...")

    market_data = get_market_data()

    print(
        "Market data received."
    )

    print("")
    print("2/4 Creating market chart...")

    chart_path = create_market_chart(
        market_data
    )

    print("")
    print("3/4 Generating article with Gemini...")

    title, body = generate_article(
        market_data
    )

    print("")
    print("Article title:")
    print(title)

    print("")
    print("4/4 Publishing to Binance Square...")

    publish_to_square(
        title,
        body,
        chart_path,
    )

    print("")
    print("==========================================")
    print(" PUBLISHED SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    main()
