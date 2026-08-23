import json
import os
import subprocess
from datetime import datetime, timezone

import requests

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL = "gemini-2.5-flash"


def get_market_data():
    url = "https://api.coingecko.com/api/v3/simple/price"

    params = {
        "ids": "bitcoin,ethereum,binancecoin",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_24hr_vol": "true",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()


def generate_article(market_data):
    prompt = f"""
You are a professional crypto market analyst writing for Binance Square.

Create one original daily crypto market article using ONLY the supplied
market data.

Requirements:
- Clear English.
- Do not invent news or statistics.
- Discuss BTC, ETH and BNB.
- Explain important market observations and risks.
- Do not promise profits.
- Mention that prices can change.
- Include $BTC, $ETH and $BNB naturally.
- Include 3-5 relevant hashtags.
- Create a strong factual title.
- Around 700-1000 words.
- Return ONLY valid JSON in this format:

{{
  "title": "Article title",
  "body": "Article body"
}}

Market data:
{json.dumps(market_data, indent=2)}

Current UTC time:
{datetime.now(timezone.utc).isoformat()}
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent"
    )

    response = requests.post(
        url,
        params={"key": GEMINI_API_KEY},
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.6,
                "responseMimeType": "application/json",
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    text = result["candidates"][0]["content"]["parts"][0]["text"]

    return json.loads(text)


def publish_to_square(title, body):
    command = [
        "node",
        "scripts/post-text.mjs",
        "--title",
        title,
        "--text",
        body,
    ]

    subprocess.run(
        command,
        cwd="./square-post",
        check=True,
    )


def main():
    print("Getting market data...")

    market_data = get_market_data()

    print("Generating article with Gemini...")

    article = generate_article(market_data)

    print("Article generated:")
    print(article["title"])

    print("Publishing to Binance Square...")

    publish_to_square(
        article["title"],
        article["body"],
    )

    print("Published successfully.")


if __name__ == "__main__":
    main()
