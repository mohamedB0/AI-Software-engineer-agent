"""
Central LLM factory for all agents.

Primary provider: Groq (free tier at https://console.groq.com)
  - GROQ_API_KEY confirmed working

Secondary providers (configured in .env):
  - Google Gemini  (GOOGLE_API_KEY) - key confirmed valid, free tier quota applies
  - Qwen/DashScope (QWEN_API_KEY)   - requires a valid DashScope key from
                                      https://dashscope.aliyuncs.com

Model tier assignment:

  Reasoning tier  - PM, Architect, Code Reviewer (judgment calls)
    Groq   : llama-3.3-70b-versatile
    Google : gemini-2.0-flash

  Coding tier  - Backend Dev, Frontend Dev (full code generation)
    Groq   : llama-3.1-70b-versatile
    Google : gemini-2.0-flash

  Fast tier  - QA Test Writer (repetitive, templated)
    Groq   : llama-3.1-8b-instant
    Google : gemini-2.0-flash

To switch providers set ACTIVE_LLM_PROVIDER in .env:
    ACTIVE_LLM_PROVIDER=groq    (default)
    ACTIVE_LLM_PROVIDER=google
    ACTIVE_LLM_PROVIDER=qwen

Groq free-tier rate limits:
  llama-3.3-70b-versatile : 30 RPM
  llama-3.1-70b-versatile : 30 RPM
  llama-3.1-8b-instant    : 30 RPM

Google Gemini free-tier:
  gemini-2.0-flash        : 15 RPM / 1 million TPM

Qwen (DashScope) free-tier:
  qwen-plus               : 20 RPM / 1 million TPM (free monthly quota)
"""

from app.config import settings

# ---------------------------------------------------------------------------
# Groq model names
# ---------------------------------------------------------------------------
GROQ_REASONING_MODEL = "llama-3.3-70b-versatile"
GROQ_CODING_MODEL = "llama-3.1-70b-versatile"
GROQ_FAST_MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Google Gemini model names
# ---------------------------------------------------------------------------
GOOGLE_MODEL = "gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Qwen (DashScope) model names
# ---------------------------------------------------------------------------
QWEN_REASONING_MODEL = "qwen-plus"
QWEN_CODING_MODEL = "qwen-plus"
QWEN_FAST_MODEL = "qwen-turbo"


def _get_provider() -> str:
    """Read ACTIVE_LLM_PROVIDER from env, default to groq."""
    return getattr(settings, "active_llm_provider", "groq").lower()


def _groq(model: str):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, api_key=settings.groq_api_key, temperature=0)


def _google(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def _qwen(model: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=settings.qwen_api_key,
        # International endpoint used by qwencloud.com accounts
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )


def _build(groq_model: str, google_model: str, qwen_model: str):
    """Return the appropriate LLM based on ACTIVE_LLM_PROVIDER."""
    provider = _get_provider()
    if provider == "google":
        return _google(google_model)
    if provider == "qwen":
        return _qwen(qwen_model)
    # Default: groq
    return _groq(groq_model)


def get_reasoning_llm():
    """
    Strong reasoning model for PM, Architect, and Code Reviewer agents.
    These make judgment calls that affect everything downstream.
    """
    return _build(GROQ_REASONING_MODEL, GOOGLE_MODEL, QWEN_REASONING_MODEL)


def get_coding_llm():
    """
    High-quality coding model for Backend and Frontend Developer agents.
    Must produce complete, runnable, correctly-typed code.
    """
    return _build(GROQ_CODING_MODEL, GOOGLE_MODEL, QWEN_CODING_MODEL)


def get_fast_llm():
    """
    Fast, lightweight model for repetitive tasks like test generation.
    """
    return _build(GROQ_FAST_MODEL, GOOGLE_MODEL, QWEN_FAST_MODEL)
