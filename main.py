import os
import json
import httpx
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Accounting Agent")

# ── Models ────────────────────────────────────────────────────────────────────

class SolveRequest(BaseModel):
    prompt: str
    proxy_url: str          # e.g. "https://proxy.ainm.no/tripletex"
    session_token: str      # Tripletex session token
    company_id: int = 0     # Tripletex companyId (0 = your own company)
    attachments: Optional[list] = None  # [{filename, base64, mime_type}]

class SolveResponse(BaseModel):
    status: str             # "completed" or "failed"
    message: Optional[str] = None

# ── Tripletex API client ───────────────────────────────────────────────────────

class TripletexClient:
    def __init__(self, proxy_url: str, session_token: str, company_id: int = 0):
        self.base_url = proxy_url.rstrip("/")
        self.company_id = company_id
        self.auth = ("0", session_token)
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def get(self, path: str, params: dict = None) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(self._url(path), headers=self.headers, auth=self.auth, params=params)
            r.raise_for_status()
            return r.json()

    async def post(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(self._url(path), headers=self.headers, auth=self.auth, json=body)
            r.raise_for_status()
            return r.json()

    async def put(self, path: str, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.put(self._url(path), headers=self.headers, auth=self.auth, json=body)
            r.raise_for_status()
            return r.json()

    async def delete(self, path: str) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.delete(self._url(path), headers=self.headers, auth=self.auth)
            if r.status_code == 204:
                return {"success": True}
            r.raise_for_status()
            return r.json()

# ── Claude agent tools (what Claude can call) ─────────────────────────────────

def build_tools() -> list:
    return [
        {
            "name": "tripletex_get",
            "description": "Perform a GET request to the Tripletex API. Use this to look up employees, customers, products, invoices, projects, departments, etc.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "API path, e.g. /employee, /customer, /invoice"},
                    "params": {"type": "object", "description": "Optional query parameters", "default": {}}
                },
                "required": ["path"]
            }
        },
        {
            "name": "tripletex_post",
            "description": "Perform a POST request to the Tripletex API to CREATE a new resource (employee, customer, product, invoice, project, department, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "API path, e.g. /employee, /customer, /invoice"},
                    "body": {"type": "object", "description": "Request body as JSON object"}
                },
                "required": ["path", "body"]
            }
        },
        {
            "name": "tripletex_put",
            "description": "Perform a PUT request to the Tripletex API to UPDATE an existing resource. Path must include the resource ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "API path including ID, e.g. /employee/123"},
                    "body": {"type": "object", "description": "Full updated resource as JSON object"}
                },
                "required": ["path", "body"]
            }
        },
        {
            "name": "tripletex_delete",
            "description": "Perform a DELETE request to the Tripletex API to remove a resource. Path must include the resource ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "API path including ID, e.g. /travelExpense/123"}
                },
                "required": ["path"]
            }
        }
    ]

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert accounting agent for Tripletex (Norwegian accounting software).
You receive accounting tasks in various languages (Norwegian, English, Spanish, Portuguese, Nynorsk, German, French) and complete them by calling the Tripletex REST API.

## Your approach
1. Read the task carefully — identify WHAT needs to be done and WHAT data is provided
2. Plan the minimum number of API calls needed (efficiency is scored)
3. Execute the plan step by step, checking results before continuing
4. For multi-step tasks, create resources in the correct dependency order

## Key Tripletex API paths
- Employees:      GET/POST /employee,         PUT /employee/{id}
- Customers:      GET/POST /customer,         PUT /customer/{id}
- Products:       GET/POST /product,          PUT /product/{id}
- Invoices:       GET/POST /invoice,          PUT /invoice/{id}
  - Send invoice:   PUT /invoice/{id}/:send
  - Payment:        POST /invoice/{id}/payment or POST /ledger/voucher
  - Credit note:    POST /invoice/{id}/:createCreditNote
- Travel expense: GET/POST /travelExpense,    DELETE /travelExpense/{id}
- Projects:       GET/POST /project,          PUT /project/{id}
- Departments:    GET/POST /department,       PUT /department/{id}
- Accounts:       GET /ledger/account
- Vouchers:       POST /ledger/voucher

## Important rules
- Always use the tools — never make up data or pretend to call APIs
- When creating an invoice, you need: customerId, date, dueDate, and at least one order line with productId (or description), quantity, unitPriceExcludingVatCurrency
- Dates must be in format YYYY-MM-DD
- Norwegian VAT code for standard goods/services: 3 (25%)
- When looking up resources, use search params like ?firstName=X or ?name=X to find by name
- For employee phone/mobile: use the phoneNumberMobile field
- Currency codes: NOK, EUR, USD, etc.
- When task mentions "registering payment" on an invoice, use POST /invoice/{id}/payment with amount and paymentDate
- Always GET to confirm a resource exists before trying to update/delete it

## Efficiency
- Minimize total API calls — only GET data you actually need
- Combine data in one request where possible (use query params to filter)
- Don't make exploratory calls unless truly necessary

When you are done, say DONE. Do not ask for confirmation — just complete the task."""

# ── Agent loop ─────────────────────────────────────────────────────────────────

async def run_agent(prompt: str, client: TripletexClient, attachments: list = None) -> str:
    """Agentic loop: Claude reasons and calls tools until task is complete."""
    anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tools = build_tools()

    # Build initial user message
    user_content = [{"type": "text", "text": prompt}]

    # Attach files if present (images/PDFs as base64)
    if attachments:
        for att in attachments:
            mime = att.get("mime_type", "image/png")
            if mime == "application/pdf":
                user_content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": att["base64"]
                    }
                })
            elif mime.startswith("image/"):
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": att["base64"]
                    }
                })

    messages = [{"role": "user", "content": user_content}]

    max_iterations = 30
    for iteration in range(max_iterations):
        logger.info(f"Agent iteration {iteration + 1}")

        response = await anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Add assistant response to message history
        messages.append({"role": "assistant", "content": response.content})

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    logger.info(f"Agent finished: {block.text[:200]}")
            return "completed"

        if response.stop_reason != "tool_use":
            logger.warning(f"Unexpected stop reason: {response.stop_reason}")
            return "completed"

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            logger.info(f"Tool call: {tool_name} {tool_input.get('path', '')}")

            try:
                if tool_name == "tripletex_get":
                    result = await client.get(tool_input["path"], tool_input.get("params", {}))
                elif tool_name == "tripletex_post":
                    result = await client.post(tool_input["path"], tool_input["body"])
                elif tool_name == "tripletex_put":
                    result = await client.put(tool_input["path"], tool_input["body"])
                elif tool_name == "tripletex_delete":
                    result = await client.delete(tool_input["path"])
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result)
                })

            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                logger.error(f"API error {e.response.status_code}: {error_body}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": True,
                    "content": f"HTTP {e.response.status_code}: {error_body}"
                })
            except Exception as e:
                logger.error(f"Tool error: {e}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": True,
                    "content": str(e)
                })

        messages.append({"role": "user", "content": tool_results})

    logger.warning("Max iterations reached")
    return "completed"

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest):
    """Main endpoint — receives task, runs agent, returns status."""
    logger.info(f"Received task: {request.prompt[:100]}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    tripletex = TripletexClient(
        proxy_url=request.proxy_url,
        session_token=request.session_token,
        company_id=request.company_id,
    )

    status = await run_agent(
        prompt=request.prompt,
        client=tripletex,
        attachments=request.attachments,
    )

    return SolveResponse(status=status)

@app.get("/health")
async def health():
    return {"status": "ok"}
