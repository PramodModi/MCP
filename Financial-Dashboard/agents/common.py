"""
agents/common.py  —  Shared utilities for all agents
=====================================================
Provides:
  • CONFIG          — loaded once from agent_config.yaml
  • get_llm()       — LLM factory (mistral / gemini / deepseek / groq)
  • build_mcp_config() — MultiServerMCPClient config dict builder (with env support)
"""

import os
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv(find_dotenv())


# =============================================================================
# CONFIG LOADER
# =============================================================================

def _find_config() -> str:
    """Walk up from cwd to find agent_config.yaml."""
    from pathlib import Path
    here = Path.cwd()
    for _ in range(4):
        candidate = here / "agent_config.yaml"
        if candidate.exists():
            return str(candidate)
        here = here.parent
    raise FileNotFoundError("agent_config.yaml not found in current or parent directories")


with open(_find_config()) as _f:
    CONFIG: dict = yaml.safe_load(_f)


# =============================================================================
# LLM FACTORY
# =============================================================================

def get_llm(size: str = "model") -> BaseChatModel:
    """
    Return a LangChain chat model based on agent_config.yaml → models.

    Args:
        size: config key to read the model name from (default "model").
              Pass "large" / "small" if those keys exist in config.

    Supported providers (models.provider):
      mistral  — ChatMistralAI,            env: MISTRAL_API_KEY
      gemini   — ChatGoogleGenerativeAI,   env: GEMINI_API_KEY
      deepseek — ChatOpenAI (compat),      env: DEEPSEEK_API_KEY
      groq     — ChatGroq,                 env: GROK_API_KEY
    """
    cfg         = CONFIG["models"]
    provider    = cfg.get("provider", "mistral").lower()
    model       = cfg.get(size, cfg.get("model", "mistral-small-latest"))
    temperature = cfg.get("temperature", 0)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise EnvironmentError("MISTRAL_API_KEY not set. Add it to .env")
        return ChatMistralAI(model=model, api_key=api_key, temperature=temperature)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set. Add it to .env")
        return ChatGoogleGenerativeAI(
            model=model, google_api_key=api_key, temperature=temperature
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not set. Add it to .env")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=temperature,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROK_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GROK_API_KEY not set. Add it to .env")
        return ChatGroq(model=model, api_key=api_key, temperature=temperature)

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: mistral | gemini | deepseek | groq"
        )


# =============================================================================
# MCP SERVER CONFIG BUILDER
# =============================================================================

def build_mcp_config(agent_name: str) -> dict:
    """
    Build a MultiServerMCPClient config dict for the given agent.

    Reads which servers the agent uses from agent_config.yaml:
      agent_servers.<agent_name> → list of server keys
      mcp_servers.<key>          → transport, command, args, env, enabled

    Env values from mcp_servers.<key>.env are merged on top of os.environ
    so that static config values (e.g. ENABLED_CAPABILITIES) override
    environment but dynamic secrets (CLIENT_ID, REFRESH_TOKEN) come from .env.

    Returns:
        Dict ready to pass to MultiServerMCPClient(config).
    """
    all_servers       = CONFIG.get("mcp_servers", {})
    agent_server_keys = CONFIG.get("agent_servers", {}).get(agent_name, [])
    mcp_config        = {}

    for key in agent_server_keys:
        server = all_servers.get(key)
        if not server:
            raise ValueError(f"Server '{key}' not found in agent_config.yaml mcp_servers")
        if not server.get("enabled", True):
            print(f"  [config] MCP server '{key}' is disabled — skipping")
            continue

        transport = server["transport"]

        if transport == "stdio":
            cmd = shutil.which(server["command"])
            if not cmd:
                # Fallback: look in the venv's bin dir alongside the running Python
                venv_bin = Path(sys.executable).parent / server["command"]
                if venv_bin.exists():
                    cmd = str(venv_bin)
            if not cmd:
                raise RuntimeError(
                    f"Command '{server['command']}' not found in PATH or venv bin.\n"
                    f"Install with: pip install google-workspace-mcp"
                )
            # Inherit process env, then overlay static values from config
            env = dict(os.environ)
            for k, v in server.get("env", {}).items():
                env[k] = str(v)

            mcp_config[key] = {
                "transport": "stdio",
                "command":   cmd,
                "args":      server.get("args", []),
                "env":       env,
            }

        elif transport == "http":
            mcp_config[key] = {
                "transport": "http",
                "url":       server["url"],
            }

        else:
            raise ValueError(f"Unknown transport '{transport}' for server '{key}'")

    return mcp_config


