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
- Suppliers:      GET/POST /supplier,         PUT /supplier/{id}
- Products:       GET/POST /product,          PUT /product/{id}
- Orders:         GET/POST /order,            PUT /order/{id}
  - Add line:       POST /order/orderline
  - To invoice:     PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD
- Invoices:       GET /invoice,              PUT /invoice/{id}
  - Send invoice:   PUT /invoice/{id}/:send?sendType=EMAIL (sendType required: EMAIL, EHF, AVTALEGIRO, or PAPER)
  - Payment:        PUT /invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=1&paidAmount=X (NOT POST, query params)
  - Credit note:    PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD (NOT POST)
  - Search:         GET /invoice requires invoiceDateFrom and invoiceDateTo params; invoiceDateTo must be at least 1 day AFTER invoiceDateFrom
  - Valid fields:   id, invoiceNumber, invoiceDate, invoiceDueDate, amount, amountExcludingVat, amountOutstanding, amountCurrency, customer, comment (NOT "status")
- Travel expense: GET/POST /travelExpense,    DELETE /travelExpense/{id}
- Projects:       GET/POST /project,          PUT /project/{id}
- Departments:    GET/POST /department,       PUT /department/{id}
- Accounts:       GET /ledger/account
- Vouchers:       POST /ledger/voucher (fields: date, description, postings[{account,customer/supplier,amount,description}])
- Free dimensions: GET/POST /ledger/accountingDimensionName, GET/POST /ledger/accountingDimensionValue
- Salary/payroll:  GET/POST /salary/payslip, GET/POST /salary/transaction

## How to create an invoice
Two valid flows:

**Flow A — via order (preferred):**
1. POST /order: {"customer": {"id": X}, "orderDate": "YYYY-MM-DD", "deliveryDate": "YYYY-MM-DD"}
2. POST /order/orderline: {"order": {"id": <order_id>}, "description": "...", "count": 1, "unitPriceExcludingVatCurrency": X, "vatType": {"id": 3}}
3. POST /invoice: {"invoiceDate": "YYYY-MM-DD", "invoiceDueDate": "YYYY-MM-DD", "customer": {"id": X}, "orders": [{"id": <order_id>}]}

**Flow B — convert order directly:**
1. POST /order (same as above)
2. POST /order/orderline (same as above)
3. PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD

IMPORTANT field names for invoice: use "invoiceDate" and "invoiceDueDate" — NOT "dueDate". Do not include extra fields like "project", "invoiceOnAccountVatHigh", etc.

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

**Supplier** (POST /supplier):
- name (required)
- isSupplier: true (always include)
- organizationNumber (if provided)
- email, phoneNumber (if provided)
- IMPORTANT: use /supplier for "leverandør/lieferant/proveedor/fornecedor/supplier" — NEVER use /customer for a supplier

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

**Payment** (PUT /invoice/{id}/:payment):
- Use PUT with QUERY PARAMS: ?paymentDate=YYYY-MM-DD&paymentTypeId=1&paidAmount=X
- paymentTypeId 1 = bank transfer (default)
- paidAmount = the amount being paid (full outstanding amount unless partial)

**Free accounting dimension** (POST /ledger/accountingDimensionName):
- dimensionName: "string" (the name, e.g. "Prosjekttype")
- active: true
- Tripletex supports exactly 3 dimension slots (index 1, 2, 3) — check existing ones first with GET /ledger/accountingDimensionName to find which dimensionIndex is assigned

**Dimension value** (POST /ledger/accountingDimensionValue):
- dimensionIndex: 1, 2, or 3 (must match the dimensionIndex of the parent dimension)
- displayName: "string" (the value name, e.g. "Internt")
- active: true
- showInVoucherRegistration: true

**Referencing dimension values in voucher postings:**
Use freeAccountingDimension1, freeAccountingDimension2, or freeAccountingDimension3 in each posting:
{"freeAccountingDimension1": {"id": <value_id>}}
The number (1/2/3) must match the dimensionIndex of the value.

**Project** (POST /project):
- name (required)
- number (project number, if provided)
- startDate (YYYY-MM-DD)
- customer: {"id": <id>} (if mentioned)
- projectManager: {"id": <employee_id>} (if mentioned)
- fixedprice: amount (if fixed price mentioned)
- isPriceCeiling: true (if fixed price / price ceiling mentioned)
- NOTE: if task only says "set fixed price on project", just GET the project then PUT with fixedprice — do NOT create orders or invoices

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
- You can make MULTIPLE tool calls in a single turn — do this for independent lookups (e.g. fetch customer and employee at the same time)

## How to approach each task
1. THINK first — read the full prompt and write out your plan as text: what resources need to be created/modified, in what order, what data you already have vs need to look up
2. EXECUTE the plan — make all independent lookups in parallel (multiple tool calls per turn)
3. COMPLETE dependent steps sequentially using results from previous calls

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

    max_iterations = 15
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
                if e.response.status_code == 403:
                    logger.error("Session token expired or invalid — aborting")
                    return "completed"
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
