# Financial Dashboard

An interactive Streamlit dashboard that reads your Gmail financial emails,
extracts transaction data using an LLM, and visualises your spending.

## Architecture

### Sync pipeline (3 deterministic phases — no LLM for fetching)

```
Gmail (OAuth2)
     │
     ▼
google-workspace-mcp              ← MCP server (stdio, pip install google-workspace-mcp)
     │
     ├─[1] query_gmail_emails ────► email IDs          (direct tool call, no LLM)
     │
     ├─[2] gmail_get_message_details (parallel, up to 10 concurrent)
     │       └─ HTML/plain body → strip HTML → normalise whitespace
     │
     └─[3] LLM extraction (batched, 10 at a time)
             ├─ regex pre-extracts merchant as a hint
             └─ LLM with_structured_output → TransactionData (Pydantic)
     │
     ▼
data/storage.py                   ← SQLite cache (financial_dashboard.db)
     │
     ▼
data/metrics.py                   ← pandas aggregations (KPIs, trends, categories)
     │
     ▼
ui/dashboard.py                   ← Streamlit + Plotly dashboard
```

### Key design decisions

| Concern | Approach |
|---|---|
| Email search & fetch | **Direct MCP tool calls** — deterministic, no LLM token cost |
| Parallel fetch | `asyncio.gather` with semaphore (10 concurrent) |
| Body normalisation | BeautifulSoup HTML strip + whitespace collapse — handles both HTML and whitespace-heavy plain-text (Axis Bank, HDFC, etc.) |
| Merchant extraction | Regex pre-extraction injects `Pre-detected Merchant: X ←` hint before LLM sees the email |
| Structured output | `llm.with_structured_output(TransactionData)` — Pydantic-validated, no JSON parsing |
| LLM usage | Only for categorisation + fallback merchant — batched 10 at a time |

---

## Quick Start

### 1. Clone & set up the virtual environment

```bash
cd Financial-Dashboard
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure your LLM API key

```bash
cp .env.example .env
```

Edit `.env` and set the key for your chosen provider:

```ini
MISTRAL_API_KEY=sk-...    # default provider
```

To switch provider, edit `agent_config.yaml`:

```yaml
models:
  provider: "gemini"          # mistral | gemini | deepseek | groq
  model:    "gemini-2.0-flash"
```

### 3. Set up Google OAuth2 credentials

`google-workspace-mcp` authenticates via three environment variables — add them to `.env`:

```ini
GOOGLE_WORKSPACE_CLIENT_ID=...
GOOGLE_WORKSPACE_CLIENT_SECRET=...
GOOGLE_WORKSPACE_REFRESH_TOKEN=...
```

Follow the steps below to obtain them. This is a **one-time** setup.

#### 3.1 — Create a Google Cloud Project

1. Open [https://console.cloud.google.com](https://console.cloud.google.com).
2. Click the **project dropdown** (top-left) → **New Project**.
3. Enter a name (e.g. `financial-dashboard`) → **Create**.
4. Make sure the new project is **selected** before continuing.

#### 3.2 — Enable the Gmail API

1. Left sidebar → **APIs & Services** → **Library**.
2. Search **Gmail API** → click it → **Enable**.

#### 3.3 — Configure the OAuth Consent Screen

1. **APIs & Services** → **OAuth consent screen** → choose **External** → **Create**.
2. Fill in **App name**, **User support email**, **Developer contact email** → **Save and Continue**.
3. **Scopes** → **Add or Remove Scopes** → search `gmail.readonly` → tick  
   **Gmail API — .../auth/gmail.readonly** → **Update** → **Save and Continue**.
4. **Test users** → **+ Add Users** → add your Gmail address → **Save and Continue**.
5. **Back to Dashboard**.

#### 3.4 — Create OAuth 2.0 Client ID and download credentials.json

1. **APIs & Services** → **Credentials** → **+ Create Credentials** → **OAuth client ID**.
2. Application type → **Desktop app** → give it a name → **Create**.
3. Click **Download JSON** on the confirmation dialog.
4. Move the file into the `credentials/` folder:

```bash
mv ~/Downloads/client_secret_*.json credentials/credentials.json
```

#### 3.5 — Run the token helper script (opens a browser once)

```bash
python scripts/get_google_token.py
```

A browser tab opens — sign in with your Google account and grant Gmail read access.  
You may see **"Google hasn't verified this app"** → click **Advanced** → **Go to … (unsafe)** → **Allow**.

The script prints your three credentials and offers to append them to `.env` automatically.

> **Token expiry**: In *Testing* mode on the consent screen the refresh token is  
> valid for 7 days. To make it permanent, go to the consent screen → **Publish App**  
> (no Google review needed for personal use with your own account as a test user).

### 3a. Verify MCP tool names (recommended after package upgrades)

```bash
python agents/email_sync.py --list-tools
```

This connects to the MCP server and prints all available tools. If names differ from
defaults, update `agent_config.yaml` → `mcp_servers.gmail.tools`:

```yaml
mcp_servers:
  gmail:
    tools:
      search:    "query_gmail_emails"         # ← update if different
      get_email: "gmail_get_message_details"  # ← update if different
```

### 4. Customise Gmail query, lookback, and exclusions (optional)

Edit `agent_config.yaml`:

```yaml
gmail:
  search_query: >-
    subject:(transaction OR debit OR payment OR receipt ...)
  max_results_per_sync: 200   # emails fetched per sync run
  lookback_years: 1           # how far back full-refresh goes

transaction_exclusions:
  merchants:
    - "Mummy"      # case-insensitive substring — excluded from dashboard
    - "Papa"
```

### 5. Run the dashboard

```bash
streamlit run ui/dashboard.py
```

Open <http://localhost:8501> in your browser.  
Click **🔄 Sync** in the sidebar to fetch emails for the first time.

---

## CLI Usage

```bash
# Incremental sync (only emails since last sync)
python agents/email_sync.py

# Full refresh (clears DB, re-fetches all emails within lookback_years)
python agents/email_sync.py --full-refresh

# List all tools exposed by the MCP server
python agents/email_sync.py --list-tools
```

---

## Dashboard Features

| Feature | Description |
|---|---|
| **Period Selector** | This Month / Last Month / Last 2 Months / Last Quarter / Last 6 Months / YTD / Custom |
| **Category Filter** | Multi-select to show/hide specific expense categories |
| **Bank Filter** | Multi-select by bank or payment app |
| **KPI Cards** | Total Spent, # Transactions, Avg Transaction, Largest Transaction |
| **Daily Trend** | Area chart of daily spend |
| **Category Donut** | Breakdown of spend by category |
| **Top Merchants** | Horizontal bar — top 10 merchants by spend |
| **Month-over-Month** | Bar chart comparing monthly totals |
| **Bank / App Bar** | Spend split by bank or payment app |
| **Transaction Table** | Full sortable table of all transactions |
| **🔄 Sync** | Incremental sync — fetches only emails since last sync |
| **🗑️ Full Refresh** | Wipes DB and re-fetches everything within the lookback window |

---

## Supported LLM Providers

| Provider | Config value | Env var |
|---|---|---|
| Mistral (default) | `mistral` | `MISTRAL_API_KEY` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Groq | `groq` | `GROK_API_KEY` |

---

## Project Structure

```
Financial-Dashboard/
├── agents/
│   ├── common.py              # LLM factory, MCP config builder
│   ├── email_sync.py          # Direct MCP calls, parallel fetch, batched LLM parse
│   └── transaction_parser.py  # Pydantic structured output, regex merchant extraction
├── data/
│   ├── storage.py             # SQLite CRUD + sync metadata
│   └── metrics.py             # Aggregations: KPIs, trends, categories
├── scripts/
│   └── get_google_token.py    # One-time OAuth2 token helper
├── ui/
│   └── dashboard.py           # Streamlit + Plotly dashboard
├── credentials/               # gitignored — OAuth credentials and token
├── .env                       # gitignored — API keys and Google credentials
├── .env.example               # Template for .env
├── agent_config.yaml          # All runtime configuration
├── requirements.txt
└── README.md
```

---

## Sync Strategy

| Mode | Trigger | Behaviour |
|---|---|---|
| **Incremental** | 🔄 Sync button / CLI | Fetches only emails newer than last sync timestamp |
| **Full Refresh** | 🗑️ Full Refresh / `--full-refresh` | Wipes SQLite DB, re-fetches all emails within `lookback_years` |

Emails are deduplicated by Gmail `message_id` — already-parsed emails are skipped automatically during incremental sync.
