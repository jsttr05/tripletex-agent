# Tripletex AI Accounting Agent

FastAPI + Claude agent that solves Tripletex accounting tasks automatically.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## With Docker

```bash
docker build -t tripletex-agent .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8000:8000 tripletex-agent
```

## Deploy to Railway / Render / Fly.io

1. Push code to GitHub
2. Connect repo on Railway/Render
3. Set env var: `ANTHROPIC_API_KEY`
4. Deploy — get your HTTPS URL
5. Submit `https://your-app.railway.app` at https://app.ainm.no/submit/tripletex

## API

### POST /solve
```json
{
  "prompt": "Create employee Kari Nordmann...",
  "proxy_url": "https://proxy.ainm.no/tripletex",
  "session_token": "abc123",
  "company_id": 0,
  "attachments": null
}
```
Returns: `{"status": "completed"}`

### GET /health
Returns: `{"status": "ok"}`

## How it works

1. Platform sends POST /solve with a task prompt + Tripletex credentials
2. Claude reads the prompt and plans API calls
3. Claude calls Tripletex REST API via the provided proxy (GET/POST/PUT/DELETE)
4. When done, returns {"status": "completed"}
5. Platform verifies the result and scores your submission
