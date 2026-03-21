"""
Local test — simulates what the competition platform sends to /solve.
Run with: python test_local.py
"""
import asyncio
import httpx

BASE_URL = "http://localhost:8000"

# Example task (the platform will send something like this)
EXAMPLE_TASK = {
    "prompt": "Create a new employee named Kari Nordmann with email kari@example.com and mobile phone +4798765432. Set her job title to 'Regnskapsfører'.",
    "proxy_url": "https://kkpqfuj-amager.tripletex.dev/v2",   # from competition platform
    "session_token": "eyJ0b2tlbklkIjoyMTQ3Njg2NDQ2LCJ0b2tlbiI6ImNiN2RmZGZiLWYwNjEtNDNmOS04Mzk4LTFlN2Y1NDM5YmMzMyJ9",   # from competition platform
    "company_id": 0,
    "attachments": None
}

async def test():
    print("Testing /health ...")
    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.get(f"{BASE_URL}/health")
        print(f"  Health: {r.json()}")

        print("\nSending task to /solve ...")
        print(f"  Prompt: {EXAMPLE_TASK['prompt'][:80]}...")
        r = await client.post(f"{BASE_URL}/solve", json=EXAMPLE_TASK)
        print(f"  Status code: {r.status_code}")
        print(f"  Response: {r.json()}")

if __name__ == "__main__":
    asyncio.run(test())
