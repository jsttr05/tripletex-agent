import asyncio
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
            if r.status_code == 204 or not r.content:
                return {"success": True}
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
  - Send invoice:   PUT /invoice/{id}/:send?sendType=EMAIL (sendType required: EMAIL, EHF, AVTALEGIRO, or PAPER). Returns 204 No Content on success.
  - Payment:        Invoice MUST be sent before payment can be registered. Flow: send first, then PUT /invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=1&paidAmount=X
  - If :payment returns 404: STOP immediately — do NOT retry with different amounts, dates, or paymentTypeIds. 404 on :payment means the invoice state does not allow payment registration. Report and stop.
  - Credit note:    PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD (NOT POST)
  - Search:         GET /invoice requires invoiceDateFrom and invoiceDateTo params; invoiceDateTo must be at least 1 day AFTER invoiceDateFrom
  - To find overdue invoices: ONE call: GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2026-12-31&fields=id,invoiceNumber,invoiceDate,invoiceDueDate,amountOutstanding,customer — then filter locally for amountOutstanding > 0 and invoiceDueDate < today. Do NOT make multiple narrow date-range searches.
  - Valid fields:   id, invoiceNumber, invoiceDate, invoiceDueDate, amount, amountExcludingVat, amountOutstanding, amountCurrency, customer, comment (NOT "status")
- Timesheets:     POST /timesheet/entry — fields: employee{id}, activity{id}, project{id}, date, hours
  - Activities:   GET /activity?name=<name>&isProjectActivity=true — ALWAYS use isProjectActivity=true for project timesheets
  - Do NOT use GET /project/activity — that endpoint does not exist and returns 422
  - An activity found with isProjectActivity=false CANNOT be used on a project timesheet (returns "Aktiviteten kan ikke benyttes")
- Travel expense: GET/POST /travelExpense,    DELETE /travelExpense/{id}
- Projects:       GET/POST /project,          PUT /project/{id}
- Departments:    GET/POST /department,       PUT /department/{id}
- Accounts:       GET /ledger/account
- Vouchers:       POST /ledger/voucher — see posting rules below
- Free dimensions: GET/POST /ledger/accountingDimensionName, GET/POST /ledger/accountingDimensionValue
- Salary/payroll:  See full flow below — DO NOT use /ledger/voucher for salary tasks

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

**Employee — 2-step process:**

Step 1 — POST /employee (basic info only):
- firstName, lastName (required)
- email, phoneNumberMobile (if provided in task)
- jobTitle (if provided)
- dateOfBirth: "YYYY-MM-DD" (if provided)
- userType: "STANDARD_WITHOUT_ACCESS" (ALWAYS include this — must not be 0 or empty; use STANDARD_WITHOUT_ACCESS unless task says otherwise)
- department: {"id": <id>} — ALWAYS required. If not specified in task, GET /department first and use the first result
- roles: if task mentions "administrator" or "admin", include {"roles": [{"name": "ROLE_ADMINISTRATOR"}]}
- DO NOT include startDate, employmentPeriodStart, or any employment date in this body — those go in step 2

Step 2 — POST /employee/employment (if task mentions start date or employment details):
- employee: {"id": <employee_id>} (from step 1 response)
- startDate: "YYYY-MM-DD"
- remunerationType: "MONTHLY_WAGE" (default unless specified)
- DO NOT include employmentType — this field does not exist on Employment and causes 422

**Customer** (POST /customer):
- name (required)
- isCustomer: true (always include)
- email, phoneNumber, organizationNumber (if provided)
- Do NOT include any address fields in the POST /customer body — they will be rejected with 422
- To set address: after POST /customer succeeds, the response includes physicalAddress.id — then PUT /address/{id} with body: {"addressLine1": "...", "postalCode": "...", "city": "...", "country": {"id": 161}}
- country id 161 = Norway (default if not specified)

**Supplier** (POST /supplier):
- name (required)
- isSupplier: true (always include)
- organizationNumber (if provided)
- email, phoneNumber (if provided)
- Do NOT include address in POST body — set it via PUT /address/{id} after creation (same as customer)
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
- paidAmount = use amountOutstanding from the invoice (do NOT calculate from exchange rate or VAT — always read amountOutstanding directly from the invoice response)
- If :payment returns 404, stop — do NOT retry with different params. Report and stop.

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

## How to book a late fee / reminder fee (purregebyr / inkassogebyr)

The correct approach is to create a new invoice for the fee — do NOT attempt a standalone voucher for this.

1. Find the overdue invoice → get the customer id from it (one search call, wide date range)
2. POST /order: {"customer": {"id": <cust_id>}, "orderDate": "YYYY-MM-DD", "deliveryDate": "YYYY-MM-DD"}
3. POST /order/orderline: {"order": {"id": <order_id>}, "description": "Purregebyr", "count": 1, "unitPriceExcludingVatCurrency": <amount>, "vatType": {"id": 3}}
4. PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD

## How to create a voucher (bilag) correctly

POST /ledger/voucher body:
```
{
  "date": "YYYY-MM-DD",
  "description": "...",
  "postings": [
    {
      "date": "YYYY-MM-DD",
      "description": "...",
      "account": {"id": <account_id>},
      "amountGross": <amount>,
      "amountGrossCurrency": <amount>
    },
    ... (balancing entry)
  ]
}
```
Rules:
- `amountGross` MUST equal `amountGrossCurrency` for NOK — they must be identical numbers
- NEVER include a posting with guiRow=0 or row=0 — Tripletex auto-generates this row. Including it causes 422 "systemgenererte".
- Every posting you include must have a non-zero row index — just omit the row field entirely and let Tripletex assign it
- Postings must balance: debits equal credits (use positive for debit, negative for credit)
- Typical accounts: 1500 = Accounts receivable, 3000 = Sales income, 8070 = Interest income, 7770 = Late fee income

## How to run payroll (kjør lønn)

**NEVER use /ledger/voucher to record salary — always use the salary API.**

Step 1 — Create a salary transaction (the payroll run):
POST /salary/transaction
Body: {"date": "YYYY-MM-DD", "year": YYYY, "month": M}
- date: last day of the salary month (e.g. "2024-01-31")
- year: the 4-digit year as integer (e.g. 2024)
- month: the month number as integer 1-12 (e.g. 1 for January)
- All three fields are required — omitting year or month causes 422

Step 2 — Create a payslip for the employee within that transaction:
POST /salary/payslip
Body: {"transaction": {"id": <tx_id>}, "employee": {"id": <emp_id>}}

Step 3 — Add wage lines to the payslip:
POST /salary/payslip/{payslip_id}/wageTransaction (or /salary/specification)
Body: {"payslip": {"id": <payslip_id>}, "wageType": {"number": 100}, "amount": <gross_salary>}
- Wage type number 100 = regular monthly salary (fastlønn). Use this by default.
- amount = the gross salary amount (e.g. 44350)

Step 4 — Execute the salary run:
PUT /salary/transaction/{tx_id}/:execute

If GET /salary/transaction returns 500, skip it and go straight to POST /salary/transaction — do not give up.
If a specific step returns 500, try the next step rather than abandoning the salary flow.

## ABSOLUTE RULES — violations directly reduce your score

**VAT TYPE: NEVER look up. NEVER call any vatType endpoint.**
- vatType id for Norwegian 25% VAT is ALWAYS `{"id": 3}` — hardcoded, permanent, never changes
- Do NOT call /vatType, /product/vatType, /ledger/vatType, or any endpoint containing "vatType"
- Any call to a vatType endpoint is a guaranteed 4xx error and a score penalty

**NEVER RETRY THE SAME ENDPOINT AFTER 404.**
- A 404 means the resource or action does not exist in its current state — changing the date, amount, or paymentTypeId will NOT fix it
- If an action endpoint (e.g. :payment, :send) returns 404, stop and report — do NOT try again with different parameters

**UNRECOVERABLE ERRORS: Do not retry after these 422 errors.**
- "bank account not registered" → this blocks ALL invoice creation paths. Do NOT retry with POST /invoice, PUT /order/{id}/:invoice, or any other invoice endpoint. Report and stop immediately.
- "company setup required" type errors → report and stop immediately

**INVOICE ACTION PATHS: Only three valid actions exist on invoices.**
- Valid: `:send`, `:payment`, `:createCreditNote` — these are the ONLY valid action paths
- Do NOT guess or invent others (e.g. `:reversePayment`, `:cancel`, `:reverse` — these do not exist and will 404)
- `PUT /invoice/{id}/:createCreditNote` REQUIRES `?date=YYYY-MM-DD` as a query param — omitting it causes 422

**PUT /order/{id}/:invoice REQUIRES query params — missing them causes 422 "invoiceDate cannot be null".**
- ALWAYS use: PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD
- Both invoiceDate AND invoiceDueDate are MANDATORY — omitting either causes immediate 422
- This is the single most common avoidable error — double-check before every call to /:invoice

**TIMESHEET ENTRIES: Validate date against project startDate before posting.**
- The project GET response includes startDate — entry date MUST be on or after this date
- If the requested date is before project startDate, do NOT post — report the conflict and stop

**ORDERLINES: Post lines SEQUENTIALLY, one per turn.**
- Tripletex uses optimistic locking on orders — posting multiple orderlines in parallel causes 409 RevisionException
- Post each orderline one at a time, waiting for 201 before posting the next
- Exception: lines for DIFFERENT orders can be parallelized

## Important rules
- Always use the tools — never make up data or pretend to call APIs
- Dates must be in format YYYY-MM-DD
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
- You can make MULTIPLE tool calls in a single turn — do this for ALL independent operations (lookups AND writes)
- When fetching /ledger/account, fetch ONCE with ?fields=id,number,name&count=300 — do NOT paginate across multiple calls

## How to approach each task
1. THINK first — read the full prompt and write out your plan as text: what resources need to be created/modified, in what order, what data you already have vs need to look up
2. EXECUTE the plan — make ALL independent calls in parallel (multiple tool calls per turn); orderlines for the same order must be sequential
3. COMPLETE dependent steps sequentially using results from previous calls

## Scoring — efficiency matters critically
- Every 4xx error (400, 404, 422) DIRECTLY reduces your score bonus
- Every extra API call DIRECTLY reduces your score bonus
- Getting it right on the first attempt is essential — do NOT guess field names or endpoints
- After creating a resource, do NOT GET it again to verify — trust the 201 response
- If you are unsure of a field name, use only the ones documented above — do not invent new ones
- Only look up resources you don't already have IDs for

When you are done, say DONE. Do not ask for confirmation — just complete the task."""

# ── Agent loop ─────────────────────────────────────────────────────────────────

async def run_agent(prompt: str, client: TripletexClient, attachments: list = None) -> str:
    """Agentic loop: Claude reasons and calls tools until task is complete."""
    anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tools = build_tools()

    # Build initial user message
    user_content = [{"type": "text", "text": prompt}]

    # Attach files if present (images/PDFs as base64, CSV/text decoded inline)
    if attachments:
        import base64
        for att in attachments:
            mime = att.get("mime_type", "application/octet-stream")
            name = att.get("name", att.get("filename", "file"))
            b64 = att.get("base64", att.get("data", ""))
            if mime == "application/pdf":
                user_content.append({
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64
                    }
                })
            elif mime.startswith("image/"):
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64
                    }
                })
            else:
                # CSV, plain text, and other text-based files — decode and embed as text
                try:
                    decoded = base64.b64decode(b64).decode("utf-8", errors="replace")
                except Exception:
                    decoded = b64  # already plain text
                user_content.append({
                    "type": "text",
                    "text": f"[Attached file: {name}]\n{decoded}"
                })
                logger.info(f"Attached text file '{name}' ({len(decoded)} chars)")

    messages = [{"role": "user", "content": user_content}]

    max_iterations = 25
    consecutive_errors: dict[str, int] = {}  # path → consecutive 4xx count

    for iteration in range(max_iterations):
        logger.info(f"Agent iteration {iteration + 1}")

        try:
            response = await anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic rate limit hit: {e}")
            return "failed"

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

        # Process tool calls — run all in parallel
        tool_blocks = [b for b in response.content if b.type == "tool_use"]

        async def execute_tool(block) -> dict:
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
                result_str = json.dumps(result)
                # Truncate very large responses to prevent context token explosion.
                # Account/product lists can be 100+ entries; keep enough to be useful.
                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "\n... [truncated, use ?fields= and filters to narrow results]"
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str
                }
            except httpx.HTTPStatusError as e:
                error_body = e.response.text
                logger.error(f"API error {e.response.status_code}: {error_body}")
                if e.response.status_code == 403 and "Invalid or expired token" in error_body:
                    raise RuntimeError("token_expired")
                path = tool_input.get("path", "")
                consecutive_errors[path] = consecutive_errors.get(path, 0) + 1
                extra = ""
                if consecutive_errors[path] >= 2:
                    extra = f" [WARNING: this is error #{consecutive_errors[path]} on {path} — stop retrying this path with the same approach, use only documented field names or report failure]"
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "is_error": True,
                    "content": f"HTTP {e.response.status_code}: {error_body}{extra}"
                }
            except Exception as e:
                logger.error(f"Tool error: {e}")
                raise

        try:
            tool_results = await asyncio.gather(*[execute_tool(b) for b in tool_blocks])
        except RuntimeError as e:
            if str(e) == "token_expired":
                logger.error("Session token expired or invalid — aborting")
                return "completed"
            raise
        except anthropic.RateLimitError as e:
            logger.error(f"Anthropic rate limit hit: {e}")
            return "failed"

        messages.append({"role": "user", "content": list(tool_results)})

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
