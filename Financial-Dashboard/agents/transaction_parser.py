"""
agents/transaction_parser.py — LLM-based Transaction Extractor
===============================================================
Uses the configured LLM to extract structured financial transaction data
from raw email content.

LLM is obtained via get_llm() from agents.common (provider set in agent_config.yaml).
API key is read from .env (MISTRAL_API_KEY, GEMINI_API_KEY, etc.).

Returns a TransactionData dict or None if:
  - The email is not a genuine financial transaction
  - LLM confidence is below 0.5
  - Extracted amount is zero or negative
"""

import re
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.common import get_llm


# =============================================================================
# Structured Output Schema
# =============================================================================

class TransactionData(BaseModel):
    is_financial: bool = Field(
        description=(
            "True if this email records an actual financial transaction: "
            "debit, payment, purchase, UPI transfer, spent, order with a paid amount, "
            "bank/service/convenience fee or charge, EMI deduction, or any money movement. "
            "False for: promotional emails, OTP emails, account statements with no new, fee acknowledgement "
            "transaction, credit card bill reminders, shipping/delivery status with no amount, marketing."
        )
    )
    date: str = Field(
        description="Transaction date in YYYY-MM-DD format. Use the email's date header if no date found in body."
    )
    amount: float = Field(
        description="Transaction amount in INR as a plain number (e.g. 1250.50). 0.0 if not found."
    )
    merchant: str = Field(
        description=(
            "Merchant, vendor, or recipient name. "
            "For bank debit alerts look for 'towards MERCHANT', 'at MERCHANT', or 'to MERCHANT' in the body. "
            "For UPI use the UPI name/VPA. "
            "For bank transfers use the beneficiary name. "
            "Include e-commerce name (e.g. AMAZONIN, MYNTRA, FLIPKART) if mentioned. "
            "'Unknown' only if truly no merchant name is present."
        )
    )
    category: str = Field(
        description=(
            "Expense category — for example: "
            "food, travel, shopping, utilities, entertainment, healthcare, "
            "education, subscription, fuel, rent, transfer, fees, other"
        )
    )
    transaction_type: str = Field(
        description="'debit' if money left the account, 'credit' if money was received."
    )
    bank_or_source: str = Field(
        description=(
            "Name of the bank or app that sent the email "
            "(e.g. HDFC Bank, ICICI Bank, Google Pay, PhonePe, Paytm, Amazon). "
            "'Unknown' if not identifiable."
        )
    )
    confidence: float = Field(
        description="Your confidence in this extraction from 0.0 (guessing) to 1.0 (certain)."
    )


# =============================================================================
# Regex merchant pre-extraction (fastest path — no LLM needed)
# =============================================================================

_RE_MERCHANT = [
    re.compile(r'\btowards\s+([A-Z][A-Z0-9 \-\.&]{1,39})', re.MULTILINE),
    re.compile(r'Merchant(?:\s+Name)?:\s*\n?\s*([^\n]{2,50})', re.IGNORECASE),
    re.compile(r'\b(?:paid|sent)\s+to\s+([A-Za-z][A-Za-z0-9 \-\.&@]{1,49})', re.IGNORECASE),
    re.compile(r'\bdebited\b.*?\bat\s+([A-Z][A-Z0-9 \-\.&]{2,39})', re.DOTALL | re.IGNORECASE),
    re.compile(r'Beneficiary(?:\s+Name)?:\s*([^\n]{2,50})', re.IGNORECASE),
]


def _extract_merchant_hint(body: str) -> str:
    """Return first regex-matched merchant name, or empty string."""
    if not body:
        return ""
    for pattern in _RE_MERCHANT:
        m = pattern.search(body)
        if m:
            return m.group(1).strip()
    return ""


# =============================================================================
# Extraction Prompt
# =============================================================================

_EXTRACTION_PROMPT = """\
You are a financial transaction extractor. Analyze the email below and extract transaction details.

Email Subject : {subject}
Email From    : {from_addr}
Email Date    : {date}
{merchant_hint_line}
Email Body    : {body}

Rules:
1. Set is_financial=true for: money debited/paid/transferred/spent, purchases (Amazon, Myntra, Flipkart,
   zomato, swiggy, zepto, any e-commerce), bank/service/convenience fees or charges, EMI deductions, UPI payments.
   Set is_financial=false ONLY for: promotions, OTPs, account statements with no new transaction,
   bill reminders with no payment confirmation, pure shipping/delivery status with no amount, marketing.
2. Amount must be in INR (₹). Convert if needed. Return as plain float (no symbols).
3. Merchant extraction — use the FIRST matching pattern found in the body:
   a) 'debited ... towards MERCHANT'           →  merchant = MERCHANT
   b) 'debited ... at MERCHANT'                →  merchant = MERCHANT
   c) 'paid to MERCHANT'                       →  merchant = MERCHANT
   d) 'sent to MERCHANT'                       →  merchant = MERCHANT
   e) Labeled field 'Merchant Name: MERCHANT'
      or 'Merchant Name:\nMERCHANT'            →  merchant = MERCHANT
   f) Labeled field 'Merchant: MERCHANT'       →  merchant = MERCHANT
   g) Labeled field 'Beneficiary Name: NAME'   →  merchant = NAME
   h) UPI VPA (name@bank format)               →  merchant = the VPA name part
   i) Beneficiary name in transfer             →  merchant = beneficiary name
   Do NOT return 'Unknown' if any of the above patterns match.
   Example:
    body : "Here's the summary of your Axis Bank Credit Card Transaction: Transaction Amount: "
   If merchant is ASSPL then consider it AMAZONIN
4. Category must be exactly one of:
   food | restaurant | hotel | travel | shopping | utilities | entertainment | healthcare |
   education | subscription | fuel | rent | transfer | fees | other
5. date must be strictly YYYY-MM-DD. Infer from email Date header if body has no date.
6. transaction_type: "debit" = money went out; "credit" = money came in.
7. Set confidence=1.0 only if all fields are clearly stated in the email.
8. If merchant is ASSPL then consider it AMAZONIN    

Return ONLY valid JSON — no explanation, no markdown fences.\
"""


# =============================================================================
# Parser
# =============================================================================

_llm = None  # Lazy-initialised once per process


async def parse_transaction(email_data: dict) -> dict | None:
    """
    Extract financial transaction data from a single email using LLM.

    Args:
        email_data: Dict with keys: id, subject, from, date, body

    Returns:
        Transaction dict matching TransactionData fields, or None if:
          - Email is not a financial transaction (is_financial=False)
          - Confidence < 0.5
          - Amount <= 0
          - LLM call fails
    """
    global _llm
    if _llm is None:
        _llm = get_llm()

    structured_llm = _llm.with_structured_output(TransactionData)

    body          = (email_data.get("body", "") or "")[:3000]
    merchant_hint = _extract_merchant_hint(body)

    def _esc(s: str) -> str:
        return str(s).replace("{", "{{").replace("}", "}}")

    hint_line = (
        f"Pre-detected Merchant (regex): {merchant_hint}  "
        "\u2190 use this as merchant value unless clearly wrong"
        if merchant_hint else ""
    )

    prompt = _EXTRACTION_PROMPT.format(
        subject           = _esc(email_data.get("subject", "")),
        from_addr         = _esc(email_data.get("from", "")),
        date              = _esc(email_data.get("date", "")),
        merchant_hint_line = hint_line,
        body              = _esc(body),
    )

    try:
        result: TransactionData = await structured_llm.ainvoke(prompt)
    except Exception:
        return None

    if result is None:
        return None
    if not result.is_financial:
        return None
    if result.confidence < 0.5:
        return None
    if result.amount <= 0:
        return None

    return result.model_dump()
