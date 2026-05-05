"""
scripts/get_google_token.py
===========================
One-time helper to obtain a Google OAuth2 refresh token for google-workspace-mcp.

Prerequisites:
    pip install google-auth-oauthlib

How to get credentials.json:
    Google Cloud Console → APIs & Services → Credentials
    → Create → OAuth 2.0 Client ID → Desktop app → Download JSON
    → Save as:  credentials/credentials.json

Usage:
    python scripts/get_google_token.py

Output:
    Prints the three lines to paste into your .env file.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT    = Path(__file__).parent.parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials" / "credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]


def main() -> None:
    if not CREDENTIALS_FILE.exists():
        print(f"ERROR: credentials.json not found at:\n  {CREDENTIALS_FILE}")
        print()
        print("Steps:")
        print("  1. Go to https://console.cloud.google.com")
        print("  2. APIs & Services → Credentials → + Create Credentials")
        print("  3. OAuth 2.0 Client ID → Desktop app → Create")
        print("  4. Download JSON → rename to credentials.json")
        print(f"  5. Move it to: {CREDENTIALS_FILE}")
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("ERROR: google-auth-oauthlib is not installed.")
        print("Install it:  pip install google-auth-oauthlib")
        sys.exit(1)

    print("Starting OAuth2 flow...")
    print("A browser window will open — sign in and grant Gmail read access.")
    print()

    flow  = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)

    client_id     = creds.client_id
    client_secret = creds.client_secret
    refresh_token = creds.refresh_token

    if not refresh_token:
        print()
        print("ERROR: No refresh_token returned.")
        print("This can happen if you previously authorised this app.")
        print("Go to https://myaccount.google.com/permissions, revoke access,")
        print("then run this script again.")
        sys.exit(1)

    env_block = (
        f"GOOGLE_WORKSPACE_CLIENT_ID={client_id}\n"
        f"GOOGLE_WORKSPACE_CLIENT_SECRET={client_secret}\n"
        f"GOOGLE_WORKSPACE_REFRESH_TOKEN={refresh_token}\n"
    )

    print()
    print("=" * 60)
    print("Success! Paste these into your .env file:")
    print("=" * 60)
    print(env_block)

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        answer = input("Append automatically to .env? [y/N]: ").strip().lower()
        if answer == "y":
            with open(env_file, "a") as f:
                f.write("\n# Google Workspace OAuth2 credentials\n")
                f.write(env_block)
            print(f"Written to {env_file}")
    else:
        answer = input("Create .env with these values? [y/N]: ").strip().lower()
        if answer == "y":
            with open(env_file, "w") as f:
                f.write("# Google Workspace OAuth2 credentials\n")
                f.write(env_block)
            print(f"Created {env_file}")


if __name__ == "__main__":
    main()
