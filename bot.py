import json
import os
import subprocess
from datetime import datetime, timezone

import requests
from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-3.6-flash"

BINANCE_SQUARE_KEY = os.environ["BINANCE_SQUARE_OPENAPI_KEY"]


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

    # Binance public market-data endpoint
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
# GEMINI ARTICLE GENERATION
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

IMPORTANT RULES:

1. Do NOT invent news.
2. Do NOT invent statistics.
3. Do NOT invent events.
4. Do NOT make guaranteed price predictions.
5. Do NOT promise profits.
6. Do NOT provide "100% accurate" trading signals.
7. Clearly explain that market data is a snapshot.
8. Analyze Bitcoin, Ethereum and BNB.
9. Explain their 24-hour price movements.
10. Discuss market strength and weakness.
11. Discuss important risks.
12. Use $BTC, $ETH and $BNB naturally.
13. Add 3-5 relevant topic hashtags at the end.
14. Make the article educational and useful.
15. Use clear professional English.
16. Avoid excessive emojis.
17. Create an interesting factual title.
18. Target approximately 700-1000 words.

Return ONLY valid JSON.

Required format:

{{
    "title": "Article title",
    "body": "Complete article body"
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
        print("Gemini returned:")
        print(text)
        raise RuntimeError(
            "Gemini response was not valid JSON."
        ) from error

    if "title" not in article:
        raise RuntimeError(
            "Gemini response does not contain title."
        )

    if "body" not in article:
        raise RuntimeError(
            "Gemini response does not contain body."
        )

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

    return {
        "title": title,
        "body": body,
    }


# ============================================================
# BINANCE SQUARE PUBLISH
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

    print("Starting Binance Square publisher...")

    result = subprocess.run(
        command,
        cwd="./square-post",
        env=env,
        text=True,
        capture_output=True,
    )

    print("Binance Square stdout:")
    print(result.stdout)

    if result.stderr:
        print("Binance Square stderr:")
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
    print("      BINANCE SQUARE AUTOMATION")
    print("==========================================")
    print("")

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print("1/3 Getting Binance market data...")

    market_data = get_market_data()

    print("Market data received successfully.")

    print(
        json.dumps(
            market_data,
            indent=2
        )
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print("")
    print("2/3 Generating article with Gemini...")

    article = generate_article(
        market_data
    )

    print("")
    print("Generated article title:")
    print("------------------------------------------")
    print(article["title"])
    print("------------------------------------------")

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print("")
    print("3/3 Publishing to Binance Square...")

    publish_to_square(
        article["title"],
        article["body"],
    )

    print("")
    print("==========================================")
    print("       PUBLISHED SUCCESSFULLY")
    print("==========================================")
    print("")


if __name__ == "__main__":
    main()
