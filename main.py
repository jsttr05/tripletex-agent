import asyncio
import os
import json
import time
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

SYSTEM_PROMPT = """You are an expert accounting agent for Tripletex (Norwegian accounting software). Complete tasks by calling the Tripletex REST API. Accept tasks in any language (NO/EN/ES/PT/DE/FR).

## Approach
1. Plan minimum API calls needed — efficiency is scored
2. Execute: make ALL independent calls in parallel; sequential only when dependent
3. THINK before writing: list what you have vs. need to look up

## API Paths
- Employees: GET/POST /employee, PUT /employee/{id}
- Customers: GET/POST /customer, PUT /customer/{id}
- Suppliers: GET/POST /supplier, PUT /supplier/{id}
- Products: GET/POST /product, PUT /product/{id}
- Orders: GET/POST /order, PUT /order/{id}; POST /order/orderline
- Invoices: GET /invoice, PUT /invoice/{id}
  - Convert order: PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD
  - Send: PUT /invoice/{id}/:send?sendType=EMAIL|EHF|AVTALEGIRO|PAPER
  - Credit note: PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD
  - Search requires invoiceDateFrom AND invoiceDateTo; invoiceDateTo must be at least 1 day after invoiceDateFrom
- Timesheets: POST /timesheet/entry (fields: employee{id}, activity{id}, project{id}, date, hours)
  - Activities: GET /activity?name=X&isProjectActivity=true — ALWAYS isProjectActivity=true for project timesheets
  - DO NOT use GET /project/activity — returns 422
- Travel expenses: GET/POST /travelExpense, DELETE /travelExpense/{id}
- Projects: GET/POST /project, PUT /project/{id}
- Departments: GET/POST /department, PUT /department/{id}
- Accounts: GET /ledger/account — fetch ONCE with ?fields=id,number,name&count=300; NEVER request again in same task
- Vouchers: POST /ledger/voucher
- Supplier invoices: POST /supplierInvoice (NEVER use /incomingInvoice — it does not exist)
- Salary: GET /salary/type, POST /salary/transaction, PUT /salary/transaction/{id}/:execute
- Free dimensions: GET/POST /ledger/accountingDimensionName, GET/POST /ledger/accountingDimensionValue
- Payment types: GET /invoice/paymentType — lists incoming payment types valid for customer invoice /:payment calls. NEVER use /ledger/paymentType or /ledger/paymentTypeOut — these return 404 or outgoing types that will be rejected.

## Required Fields Per Resource

**Employee — 2 steps:**
Step 1 POST /employee: firstName, lastName, userType: "STANDARD_WITHOUT_ACCESS", department: {"id": X}. Also: email, phoneNumberMobile, jobTitle, dateOfBirth if provided. If admin role: roles: [{"name": "ROLE_ADMINISTRATOR"}]. DO NOT include startDate here.
Step 2 POST /employee/employment (if start date given):
{"employee": {"id": X}, "startDate": "YYYY-MM-DD", "employmentDetails": [{"date": "YYYY-MM-DD", "employmentType": "ORDINARY", "remunerationType": "MONTHLY_WAGE", "employmentForm": "PERMANENT"}]}
employmentType/remunerationType MUST be inside employmentDetails[0], not on the employment body.

**Customer** POST /customer: name required; email, phoneNumber, organizationNumber if provided. No address fields in POST — after creation PUT /address/{physicalAddress.id}: {"addressLine1": "...", "postalCode": "...", "city": "...", "country": {"id": 161}}. Country 161 = Norway.

**Supplier** POST /supplier: name required; organizationNumber, email, phoneNumber if provided. No address in POST — same PUT /address/{id} pattern. Use /supplier for leverandor/proveedor/fornecedor — NEVER /customer.

**Product** POST /product: name, priceExcludingVatCurrency required; costExcludingVatCurrency, number if provided; vatType: {"id": 3}.

**Order** POST /order: customer: {"id": X}, orderDate, deliveryDate.

**Order line** POST /order/orderline: order: {"id": X}, description or product: {"id": X}, count, unitPriceExcludingVatCurrency, vatType: {"id": 3}.

**Invoice via order (preferred):**
1. POST /order; 2. POST /order/orderline; 3. PUT /order/{id}/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD
Invoice fields: invoiceDate, invoiceDueDate — NOT dueDate. Valid GET fields: id, invoiceNumber, invoiceDate, invoiceDueDate, amount, amountExcludingVat, amountOutstanding, amountCurrency, customer, comment. NOT status.

**Supplier invoice** POST /supplierInvoice (NEVER /incomingInvoice):
{"invoiceHeader": {"vendorId": X, "invoiceDate": "YYYY-MM-DD", "dueDate": "YYYY-MM-DD", "invoiceAmount": X, "invoiceNumber": "...", "description": "..."}, "orderLines": [{"row": 1, "externalId": "1", "description": "...", "accountId": X, "amountInclVat": X, "vatTypeId": X}]}
All IDs are plain integers (not objects). row starts at 1. externalId is REQUIRED on every orderLine — use the line number as a string ("1", "2", etc.) if no explicit ID exists. Optional: ?sendTo=ledger.

**Voucher** POST /ledger/voucher:
{"date": "YYYY-MM-DD", "description": "...", "postings": [{"date": "YYYY-MM-DD", "account": {"id": X}, "amount": 100.0}, {"date": "YYYY-MM-DD", "account": {"id": Y}, "amount": -100.0}]}
Postings must sum to 0. Positive = debit, negative = credit. Omit row field entirely — row 0 is system-generated and must never be included.

**Payment registration:**
1. GET /invoice/paymentType to find the correct incoming paymentTypeId
2. PUT /invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=X (body: {})
paidAmountCurrency optional. Invoice must be in sent state before payment can be registered.

**Late fee (purregebyr):**
1. GET /invoice wide range → get customer id
2. POST /order → POST /order/orderline (description: "Purregebyr", vatType: {"id": 3})
3. PUT /order/{id}/:invoice?invoiceDate=...&invoiceDueDate=...

**Salary:**
1. GET /salary/type?number=100&fields=id,number,name → get salary type id
2. POST /salary/transaction: {"date": "YYYY-MM-DD", "year": YYYY, "month": M, "payslips": [{"employee": {"id": X}, "specifications": [{"salaryType": {"id": X}, "rate": X, "count": 1, "amount": X}]}]}
3. PUT /salary/transaction/{id}/:execute
POST /salary/payslip does not exist standalone. NEVER use /ledger/voucher for salary.

**Free dimensions:**
- POST /ledger/accountingDimensionName: {"dimensionName": "...", "active": true} — check existing first; only 3 slots (index 1-3)
- POST /ledger/accountingDimensionValue: {"dimensionIndex": 1|2|3, "displayName": "...", "active": true, "showInVoucherRegistration": true}
- In voucher postings: freeAccountingDimension1/2/3: {"id": X} — number must match dimensionIndex

**Returned/reversed payment:** If a payment was returned by the bank or needs to be reversed,
do NOT use /:createCreditNote — that cancels the invoice. Instead:
1. Find the payment voucher via GET /ledger/voucher filtered by the invoice
2. Reverse it via POST /ledger/voucher with negated postings, OR
3. Check if DELETE /ledger/voucher/{id} is available for the payment entry
The invoice must remain open and show the full outstanding amount after reversal.

**Project** POST /project: name required; number, startDate, customer: {"id": X}, projectManager: {"id": X}, fixedprice, isPriceCeiling: true if fixed price. To set fixed price only: GET then PUT.

**Overdue invoices:** GET /invoice?invoiceDateFrom=2020-01-01&invoiceDateTo=2026-12-31&fields=id,invoiceNumber,invoiceDate,invoiceDueDate,amountOutstanding,customer — filter locally.

## Absolute Rules

**VAT:** vatType {"id": 3} = 25% Norwegian VAT. NEVER call any vatType endpoint — guaranteed 4xx.

**Scope guard:** Only act on what the task explicitly requests. Before every POST/PUT/DELETE: did the task ask for this? For analytical tasks (analyze/overview/report/list/compare/summarize/reconcile): GET only. Do NOT create records without explicit instruction. Do NOT send emails (/:send) unless explicitly asked.

**Account list:** You have already fetched /ledger/account at the start. It is in your context. Do NOT request it again under any circumstances. If you are about to call /ledger/account again, stop — look up the account in the data already in your conversation history and proceed.

**404 = stop:** Never retry after 404. Report and stop.

**422 unrecoverable:** bank account not registered / company setup required → stop immediately.

**Invoice actions:** :send, :createCreditNote, :payment, :createReminder. No others exist.

**Credit note:** Any reverse/cancel/credit/void/storno/kreditnota/reverser/annuller/stornieren/estornar/nota de credito → PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD. Always proceed.

**Order → invoice:** Both ?invoiceDate= AND ?invoiceDueDate= required — missing either causes 422.

**Orderlines:** Post sequentially for the same order (409 if parallel). Different orders can be parallel.

**Timesheet dates:** Entry date must be >= project startDate.

**Department:** If creating employee with no department specified, GET /department first.

## Efficiency
- Use ?fields= to limit response size
- Do NOT GET after creating — trust 201
- No speculative or exploratory calls
- Parallelize all independent calls
- Pre-flight: confirm all required fields before every write — never send speculatively

When done, say DONE."""

# ── Rate limit tracker + concurrency semaphore ─────────────────────────────────

# Configurable via env vars:
#   ANTHROPIC_RPM_LIMIT      — requests per minute ceiling (default 40)
#   ANTHROPIC_RPM_WINDOW     — rolling window in seconds (default 60)
#   ANTHROPIC_MAX_CONCURRENT — max simultaneous Anthropic API calls across all tasks (default 2)
_RPM_LIMIT       = int(os.environ.get("ANTHROPIC_RPM_LIMIT",       "40"))
_RPM_WINDOW      = int(os.environ.get("ANTHROPIC_RPM_WINDOW",      "60"))
_MAX_CONCURRENT  = int(os.environ.get("ANTHROPIC_MAX_CONCURRENT",  "2"))

# Module-level semaphore shared across all concurrent agent tasks.
# Prevents multiple simultaneous tasks from hammering the API and compounding 429 penalties.
_anthropic_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


class _RateLimitTracker:
    """Tracks Anthropic API calls in a rolling window and proactively sleeps
    before a call if we are approaching the per-minute limit, rather than
    waiting for a 429 penalty which causes 30-45s reactive delays."""

    def __init__(self, limit: int = _RPM_LIMIT, window: float = _RPM_WINDOW):
        self._limit  = limit
        self._window = window
        self._calls: list[float] = []  # timestamps of recent calls

    async def wait_if_needed(self) -> None:
        now = time.monotonic()
        # Drop timestamps outside the rolling window
        self._calls = [t for t in self._calls if now - t < self._window]
        if len(self._calls) >= self._limit - 2:  # leave 2-call safety margin
            oldest = self._calls[0]
            sleep_for = self._window - (now - oldest) + 0.5
            if sleep_for > 0:
                logger.info(f"Proactive rate-limit sleep {sleep_for:.1f}s ({len(self._calls)}/{self._limit} calls in window)")
                await asyncio.sleep(sleep_for)
        self._calls.append(time.monotonic())


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
            b64 = att.get("base64", att.get("content_base64", att.get("data", "")))
            if mime == "application/pdf":
                if not b64:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Attached PDF '{name}' is empty or was not received correctly. Please re-upload the file."
                    )
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
    # Per-run cache for /ledger/account — chart of accounts never changes mid-task.
    account_cache: dict[str, str] = {}
    rate_tracker = _RateLimitTracker()

    # FIX 4: Retry constants for Anthropic API overload (529) errors
    _ANTHROPIC_MAX_RETRIES = 3
    _ANTHROPIC_RETRY_DELAYS = [5, 15, 30]

    for iteration in range(max_iterations):
        logger.info(f"Agent iteration {iteration + 1}")

        await rate_tracker.wait_if_needed()

        # FIX 4: Wrap Anthropic API call with 529 overloaded retry logic
        response = None
        for attempt in range(_ANTHROPIC_MAX_RETRIES):
            try:
                async with _anthropic_semaphore:
                    response = await anthropic_client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=8192,
                        system=SYSTEM_PROMPT,
                        tools=tools,
                        messages=messages,
                    )
                break  # success — exit retry loop
            except anthropic.RateLimitError as e:
                logger.error(f"Anthropic rate limit hit: {e}")
                return "failed"
            except anthropic.BadRequestError as e:
                logger.error(f"Anthropic bad request: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except anthropic.APIStatusError as e:
                if e.status_code == 529:
                    if attempt < _ANTHROPIC_MAX_RETRIES - 1:
                        delay = _ANTHROPIC_RETRY_DELAYS[attempt]
                        logger.warning(f"Anthropic overloaded (529), retrying in {delay}s (attempt {attempt + 1}/{_ANTHROPIC_MAX_RETRIES})")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error("Anthropic overloaded (529) — all retries exhausted")
                        raise HTTPException(
                            status_code=503,
                            detail="Anthropic API is temporarily overloaded. Please retry your request in a moment."
                        )
                logger.error(f"Anthropic API status error {e.status_code}: {e}")
                raise HTTPException(status_code=500, detail=f"Anthropic API error {e.status_code}: {e}")
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                raise HTTPException(status_code=500, detail=f"Unexpected error calling Anthropic API: {e}")

        if response is None:
            logger.error("No response from Anthropic API after all retries")
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
            path = tool_input.get("path", "")
            logger.info(f"Tool call: {tool_name} {path}")

            # True only for the account list endpoint, not /ledger/account/{id}
            _is_account_list = (
                path.rstrip("/") == "/ledger/account"
                or (
                    path.startswith("/ledger/account")
                    and not path[len("/ledger/account"):].lstrip("/")[:1].isdigit()
                )
            )

            try:
                if tool_name == "tripletex_get":
                    # FIX 5 (strengthened): Cache the full account list.
                    # On a cache hit return a forceful warning — the polite version was being ignored.
                    if _is_account_list and "/ledger/account" in account_cache:
                        account_cache["_hit_count"] = str(int(account_cache.get("_hit_count", "0")) + 1)
                        logger.warning(f"SCOPE: agent requested /ledger/account again (hit #{account_cache['_hit_count']}) — returning warning only")
                        return {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": (
                                "[CRITICAL ERROR: You called /ledger/account again. This is forbidden. "
                                "The full account list is already in your conversation history from iteration 1. "
                                "Scroll back and find the account number you need. "
                                "Your next action must NOT be another /ledger/account call — "
                                "proceed immediately with the write operation using account data you already have.]"
                            ),
                        }
                    result = await client.get(path, tool_input.get("params", {}))

                elif tool_name == "tripletex_post":
                    body = tool_input["body"]

                    # FIX 1: Redirect /incomingInvoice → /supplierInvoice.
                    # The endpoint /incomingInvoice does not exist; /supplierInvoice is correct.
                    if path == "/incomingInvoice":
                        logger.info("REDIRECT: /incomingInvoice → /supplierInvoice")
                        path = "/supplierInvoice"

                    # Strip system-generated row 0 from voucher postings before sending.
                    # Tripletex always rejects postings with row=0 or guiRow=0 with a 422.
                    if path == "/ledger/voucher" and isinstance(body.get("postings"), list):
                        original_count = len(body["postings"])
                        body["postings"] = [
                            p for p in body["postings"]
                            if p.get("row") != 0 and p.get("guiRow") != 0
                        ]
                        stripped = original_count - len(body["postings"])
                        if stripped:
                            logger.info(f"Stripped {stripped} row-0 posting(s) from /ledger/voucher body")

                    # FIX 3: Auto-populate missing externalId on /supplierInvoice orderLines.
                    # externalId is required on every line; use line index as string if absent.
                    if path == "/supplierInvoice" and isinstance(body.get("orderLines"), list):
                        for i, line in enumerate(body["orderLines"]):
                            if not line.get("externalId"):
                                line["externalId"] = str(i + 1)
                                logger.info(f"AUTO-INJECT: externalId='{i + 1}' added to supplierInvoice orderLines[{i}]")

                    result = await client.post(path, body)

                elif tool_name == "tripletex_put":
                    put_path = path
                    put_body = tool_input["body"]

                    # FIX 2: Auto-inject sendType=EMAIL on /:send if missing.
                    # This error has appeared in every single log set — fix it unconditionally in code.
                    if "/:send" in put_path and "sendType" not in put_path:
                        logger.warning(f"AUTO-INJECT: sendType=EMAIL added to {put_path}")
                        sep = "&" if "?" in put_path else "?"
                        put_path = put_path + sep + "sendType=EMAIL"

                    result = await client.put(put_path, put_body)

                elif tool_name == "tripletex_delete":
                    result = await client.delete(path)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                result_str = json.dumps(result)
                # Truncate very large responses to prevent context token explosion.
                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "\n... [truncated, use ?fields= and filters to narrow results]"

                # Populate account cache on first successful /ledger/account list fetch.
                if tool_name == "tripletex_get" and _is_account_list and "/ledger/account" not in account_cache:
                    account_cache["/ledger/account"] = result_str
                    logger.info(f"Cached /ledger/account ({len(result_str)} chars)")

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
                consecutive_errors[path] = consecutive_errors.get(path, 0) + 1
                extra = ""
                if consecutive_errors[path] >= 2:
                    extra = (
                        f" [WARNING: this is error #{consecutive_errors[path]} on {path} — "
                        f"stop retrying this path with the same approach, use only documented field names or report failure]"
                    )
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
    # Log full prompt and attachment metadata so we can inspect tasks from container logs.
    logger.info(f"TASK PROMPT: {request.prompt}")
    attachments_meta = []
    for att in (request.files or request.attachments or []):
        attachments_meta.append({
            "name": att.get("name", att.get("filename", "?")),
            "mime_type": att.get("mime_type", att.get("type", "?")),
            "keys": list(att.keys()),
            "base64_len": len(att.get("base64", att.get("content_base64", att.get("data", ""))) or ""),
        })
    if attachments_meta:
        logger.info(f"TASK ATTACHMENTS: {attachments_meta}")
    else:
        logger.info("TASK ATTACHMENTS: none")

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

    # Complexity classifier — determines timeout (300s vs 240s).
    # Structure-based heuristics first, then multilingual keyword fallback.
    # AGENT_TIMEOUT_SECONDS env var overrides everything.
    _COMPLEX_KEYWORDS = {
        # English
        "analyse", "analyze", "reconcil", "compare", "match", "audit", "review",
        "search", "find error", "go through", "report", "errors", "date range",
        # Norwegian
        "analysere", "avstemm", "finn", "gå gjennom", "sammenlign", "feil",
        "årsoppgjør", "avskrivning", "avskrivninger", "bokfør", "bokføring",
        "regnskap", "balanse", "resultat", "årsregnskap", "depreciati",
        "bilag", "periode", "kvartal", "årlig", "søk",
        # German
        "analysieren", "vergleich", "abstimm", "prüf", "durchgeh", "fehler",
        # French
        "clôture", "amortissement", "amortissements", "comptabilisez", "comptabiliser",
        "immobilisation", "extourne", "provision", "bénéfice", "annuelle", "rapproch",
        # Portuguese
        "reconcil", "analise", "comparar", "verificar", "rever", "erros",
    }
    prompt_lower = request.prompt.lower()
    has_attachment = bool(attachments)
    # Multi-action: more than one numbered step, bullet, or semicolon-separated clause
    has_multiple_actions = (
        prompt_lower.count("\n") > 2
        or any(f"{n})" in prompt_lower for n in range(2, 6))
        or prompt_lower.count(";") >= 2
    )
    keyword_match = any(kw in prompt_lower for kw in _COMPLEX_KEYWORDS)
    is_complex = has_attachment or has_multiple_actions or keyword_match
    reason = []
    if has_attachment:        reason.append("attachment")
    if has_multiple_actions:  reason.append("multi-action")
    if keyword_match:         reason.append("keyword")
    default_timeout = 300 if is_complex else 240
    timeout_seconds = int(os.environ.get("AGENT_TIMEOUT_SECONDS", str(default_timeout)))
    logger.info(f"Agent timeout: {timeout_seconds}s (complex={is_complex}, reasons={reason})")

    try:
        status = await asyncio.wait_for(
            run_agent(
                prompt=request.prompt,
                client=tripletex,
                attachments=attachments,
            ),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error(f"run_agent timed out after {timeout_seconds}s")
        raise HTTPException(
            status_code=504,
            detail="Agent timed out. The task may be too complex or the API is under heavy load.",
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