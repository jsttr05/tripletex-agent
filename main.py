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

class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str

class SolveRequest(BaseModel):
    prompt: str
    tripletex_credentials: Optional[TripletexCredentials] = None
    # legacy fields for local testing
    proxy_url: Optional[str] = None
    session_token: Optional[str] = None
    company_id: int = 0
    attachments: Optional[list] = None
    files: Optional[list] = None  # platform uses "files" not "attachments"

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
- Orders:         GET/POST /order,            PUT /order/{id}
  - Add line:       POST /order/orderline
  - To invoice:     PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD
- Invoices:       GET /invoice,              PUT /invoice/{id}
  - Send invoice:   PUT /invoice/{id}/:send
  - Payment:        POST /invoice/{id}/payment
  - Credit note:    POST /invoice/{id}/:createCreditNote
- Travel expense: GET/POST /travelExpense,    DELETE /travelExpense/{id}
- Projects:       GET/POST /project,          PUT /project/{id}
- Departments:    GET/POST /department,       PUT /department/{id}
- Accounts:       GET /ledger/account
- Vouchers:       POST /ledger/voucher

## CRITICAL: How to create an invoice
NEVER use POST /invoice directly. The correct flow is:
1. POST /order with: {"customer": {"id": X}, "orderDate": "YYYY-MM-DD", "deliveryDate": "YYYY-MM-DD"}
2. POST /order/orderline with: {"order": {"id": <order_id>}, "description": "...", "count": 1, "unitPriceExcludingVatCurrency": X, "vatType": {"id": 3}}
3. PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD (pass dates as query params, no body needed)

## Required fields per resource type

**Employee** (POST /employee):
- firstName, lastName (required)
- email, phoneNumberMobile (if provided in task)
- jobTitle (if provided)
- roles: if task mentions "administrator" or "admin", include {"roles": [{"name": "ROLE_ADMINISTRATOR"}]}
- department: if mentioned, look up by name first and include {"department": {"id": <id>}}

**Customer** (POST /customer):
- name (required)
- isCustomer: true (always include)
- email, phoneNumber, organizationNumber (if provided)
- address if street/city/zip given: {"physicalAddress": {"addressLine1": "...", "city": "...", "postCode": "...", "country": {"id": 161}}} (161 = Norway)

**Product** (POST /product):
- name (required)
- priceExcludingVatCurrency (required, the sales price)
- costExcludingVatCurrency (if provided)
- number (product number, if provided)
- vatType: {"id": 3} for standard 25% Norwegian VAT — do NOT look this up, always use id 3

**Order** (POST /order):
- customer: {"id": <id>} (required)
- orderDate: "YYYY-MM-DD" (required)
- deliveryDate: "YYYY-MM-DD"

**Order line** (POST /order/orderline):
- order: {"id": <order_id>}
- description or product: {"id": <product_id>}
- count: number of units
- unitPriceExcludingVatCurrency: price per unit
- vatType: {"id": 3}

**Payment** (POST /invoice/{id}/payment):
- paymentDate (YYYY-MM-DD), amount, paymentTypeId: 1

**Project** (POST /project):
- name (required)
- number (project number, if provided)
- startDate (YYYY-MM-DD)
- customer: {"id": <id>} (if mentioned)
- projectManager: {"id": <employee_id>} (if mentioned)
- fixedprice: amount (if fixed price mentioned)
- isPriceCeiling: true (if fixed price / price ceiling mentioned)

## Important rules
- Always use the tools — never make up data or pretend to call APIs
- Dates must be in format YYYY-MM-DD
- Norwegian VAT id is always 3 (25%) — never call /vatType to look it up
- When looking up resources, use search params like ?firstName=X&lastName=Y or ?name=X to find by name
- Use ?fields=id,name (or relevant fields) to limit response size
- Currency codes: NOK, EUR, USD, etc. Default to NOK if not specified
- Always GET to confirm a resource exists before trying to update/delete it
- If a task mentions a role (administrator, user, etc.), always set it — it is heavily weighted in scoring
- Norwegian characters (æ, ø, å) are supported — use them as-is from the prompt

## Efficiency
- Minimize total API calls — only GET data you actually need
- Use ?fields= to fetch only needed fields
- Combine lookups where possible using query params to filter
- Do NOT make exploratory or speculative calls
- Avoid 4xx errors — plan your calls correctly before executing
- Do NOT look up vatType, currency, or other static data — use known values directly

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

async def handle_solve(request: SolveRequest) -> SolveResponse:
    logger.info(f"Received task: {request.prompt[:100]}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set")

    # Support both platform format (tripletex_credentials) and legacy format (proxy_url)
    if request.tripletex_credentials:
        proxy_url = request.tripletex_credentials.base_url
        session_token = request.tripletex_credentials.session_token
    else:
        proxy_url = request.proxy_url
        session_token = request.session_token

    if not proxy_url or not session_token:
        raise HTTPException(status_code=400, detail="Missing Tripletex credentials")

    tripletex = TripletexClient(
        proxy_url=proxy_url,
        session_token=session_token,
        company_id=request.company_id,
    )

    attachments = request.files or request.attachments

    status = await run_agent(
        prompt=request.prompt,
        client=tripletex,
        attachments=attachments,
    )

    return SolveResponse(status=status)

@app.post("/solve", response_model=SolveResponse)
async def solve(request: SolveRequest):
    return await handle_solve(request)

@app.post("/", response_model=SolveResponse)
async def solve_root(request: SolveRequest):
    return await handle_solve(request)

@app.get("/health")
async def health():
    return {"status": "ok"}
