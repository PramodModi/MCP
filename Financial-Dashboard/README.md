# Financial Dashboard

An interactive Streamlit dashboard that reads your Gmail financial emails,
extracts transaction data using an LLM, and visualises your spending.

## Architecture

```
Gmail (OAuth2)
     │
     ▼
google-workspace-mcp         ← Google's MCP server (stdio, pip install)
     │  search_emails / get_email tools
     ▼
agents/email_sync.py         ← MCP client — fetches emails
     │
     ▼
agents/transaction_parser.py ← LLM extracts amount/merchant/category
     │
     ▼
data/storage.py              ← SQLite cache (financial_dashboard.db)
     │
     ▼
data/metrics.py              ← pandas aggregations
     │
     ▼
ui/dashboard.py              ← Streamlit + Plotly dashboard
```

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

`google-workspace-mcp` authenticates via three environment variables:

```
GOOGLE_WORKSPACE_CLIENT_ID
GOOGLE_WORKSPACE_CLIENT_SECRET
GOOGLE_WORKSPACE_REFRESH_TOKEN
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
4. Move and rename the file:

```bash
mv ~/Downloads/client_secret_*.json credentials/credentials.json
```

#### 3.5 — Run the token helper script (opens a browser once)

```bash
pip install google-auth-oauthlib   # one-time dependency
python scripts/get_google_token.py
```

A browser tab opens — sign in with your Google account and grant Gmail read access.  
You may see **"Google hasn't verified this app"** → click **Advanced** → **Go to … (unsafe)** → **Allow**.

The script prints your credentials and offers to append them to `.env` automatically.

> **Token expiry**: In *Testing* mode on the consent screen the refresh token is  
> valid for 7 days. To make it permanent, go to the consent screen → **Publish App**  
> (no Google review needed for personal/test use with your own account added as a test user).

### 3a. Verify MCP tool names (recommended)

`google-workspace-mcp` tool names can vary by version. Before syncing, run:

```bash
python agents/email_sync.py --list-tools
```

This connects to the server and prints all available tools. If the names differ from
defaults, update `agent_config.yaml` → `gmail_mcp_server.tools`:

```yaml
gmail_mcp_server:
  tools:
    search_emails: "search_emails"   # ← update to match printed name
    get_email:     "get_email"        # ← update to match printed name
```

### 4. Customise Gmail query & exclusions (optional)

Edit `agent_config.yaml`:

```yaml
gmail:
  search_query: >-
    subject:(transaction OR debit OR payment OR receipt ...)
  max_results_per_sync: 200

transaction_exclusions:
  merchants:
    - "Mummy"      # transfers to exclude from the dashboard
    - "Papa"
```

### 5. Run the dashboard

```bash
streamlit run ui/dashboard.py
```

Open <http://localhost:8501> in your browser.  
Click **🔄 Sync** in the sidebar to fetch emails for the first time.

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
| **🗑️ Full Refresh** | Wipes DB and re-fetches everything from Gmail |

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
│   ├── email_sync.py          # MCP client — syncs emails → SQLite
│   └── transaction_parser.py  # LLM extraction → structured JSON
├── data/
│   ├── storage.py             # SQLite CRUD + sync metadata
│   └── metrics.py             # Aggregations: KPIs, trends, categories
├── ui/
│   └── dashboard.py           # Streamlit dashboard
├── credentials/               # gitignored — put credentials.json here
├── .env                       # gitignored — API keys
├── .env.example               # Template for .env
├── agent_config.yaml          # All runtime configuration
├── requirements.txt
└── README.md
```

---

## Cache Invalidation

- **Incremental sync (🔄 Sync)**: fetches only emails newer than the last sync timestamp. Fast and cost-efficient.
- **Full refresh (🗑️ Full Refresh)**: wipes the SQLite database and re-fetches all matching emails. Use when you change the Gmail query or want a clean slate.
- Emails are keyed by Gmail `message_id` — already-parsed emails are never re-processed during incremental sync.
