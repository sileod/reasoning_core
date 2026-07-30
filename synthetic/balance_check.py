import os
import requests
import urllib3
import certifi
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Disable SSL warnings ONLY if using verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Get API key
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

# OpenRouter endpoint
url = "https://openrouter.ai/api/v1/auth/key"

# Request headers
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

try:
    # Preferred secure request
    response = requests.get(
        url,
        headers=headers,
        verify=certifi.where(),  # uses updated CA bundle
        timeout=30
    )

except requests.exceptions.SSLError:
    print("⚠️ SSL verification failed.")
    print("Retrying insecurely (development mode only)...")

    # Fallback for corporate SSL interception / proxy environments
    response = requests.get(
        url,
        headers=headers,
        verify=False,
        timeout=30
    )

except requests.exceptions.RequestException as e:
    print("❌ Network error:")
    print(e)
    exit(1)

# Handle API response
if response.status_code != 200:
    print("❌ API Error")
    print("Status:", response.status_code)
    print("Response:", response.text)
    exit(1)

# Parse JSON
data = response.json()

# Extract usage info
usage = data.get("data", {})

print("\n🔑 OpenRouter Key Info")
print("-" * 40)

print(f"Label      : {usage.get('label')}")
print(f"Usage      : ${usage.get('usage', 0):.4f}")
print(f"Limit      : ${usage.get('limit', 'unknown')}")

if usage.get("limit") is not None:
    remaining = usage["limit"] - usage["usage"]
    print(f"Remaining  : ${remaining:.4f}")

print("-" * 40)