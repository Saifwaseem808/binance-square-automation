import json
import os
import subprocess
from datetime import datetime, timezone

import requests
from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-2.5-flash"

BINANCE_SQUARE_KEY = os.environ["BINANCE_SQUARE_OPENAPI_KEY"]


# ============================================================
# GET MARKET DATA
# ============================================================

def get_market_data():

    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
    ]

    market_data = {}

    # Binance's official market-data-only endpoint
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
# GENERATE ARTICLE WITH GEMINI
# ============================================================

def generate_article(market_data):

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    prompt = f"""
You are an experienced crypto market analyst writing for Binance Square.

Create ONE original daily crypto market article using ONLY the market
data supplied below.

IMPORTANT RULES:

1. Do not invent news.
2. Do not invent statistics.
3. Do not claim that an event happened unless it is present in the supplied data.
4. Do not promise profits.
5. Do not give guaranteed trading predictions.
6. Explain that prices are snapshots and can change.
7. Discuss Bitcoin, Ethereum and BNB.
8. Explain their current 24-hour price movements.
9. Highlight useful market observations.
10. Discuss risks.
11. Use $BTC, $ETH and $BNB naturally.
12. Add 3-5 relevant hashtags.
13. Make the article educational and useful.
14. Use clear English.
15. Target approximately 700-1000 words.
16. Create an interesting but factual title.

Return ONLY valid JSON in exactly this format:

{{
    "title": "Article title",
    "body": "Full article body"
}}

CURRENT MARKET DATA:

{json.dumps(market_data, indent=2)}

CURRENT UTC TIME:

{datetime.now(timezone.utc).isoformat()}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.6,
            response_mime_type="application/json",
        ),
    )

    text = response.text

    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    article = json.loads(text)

    if "title" not in article:
        raise RuntimeError("Gemini response has no title.")

    if "body" not in article:
        raise RuntimeError("Gemini response has no body.")

    if not article["body"].strip():
        raise RuntimeError("Generated article body is empty.")

    return article


# ============================================================
# PUBLISH TO BINANCE SQUARE
# ============================================================

def publish_to_square(title, body):

    env = os.environ.copy()

    env["BINANCE_SQUARE_OPENAPI_KEY"] = BINANCE_SQUARE_KEY

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
        check=True,
        text=True,
        capture_output=True,
    )

    print("Binance Square response:")
    print(result.stdout)

    if result.stderr:
        print("Binance Square warnings:")
        print(result.stderr)


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" Binance Square Automation")
    print("========================================")

    print("")
    print("1/3 Getting Binance market data...")

    market_data = get_market_data()

    print("Market data received successfully.")

    print("")
    print("2/3 Generating article with Gemini...")

    article = generate_article(
        market_data
    )

    print("")
    print("Generated title:")
    print(article["title"])

    print("")
    print("3/3 Publishing to Binance Square...")

    publish_to_square(
        article["title"],
        article["body"],
    )

    print("")
    print("========================================")
    print(" Successfully published to Binance Square")
    print("========================================")


if __name__ == "__main__":
    main()
