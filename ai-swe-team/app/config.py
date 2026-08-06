from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider selection: groq | google | qwen  (default: groq)
    active_llm_provider: str = "groq"

    # LLM API keys
    groq_api_key: str = ""          # primary - free tier at console.groq.com
    anthropic_api_key: str = ""     # optional fallback
    openai_api_key: str = ""        # used by ChromaDB RAG embeddings
    google_api_key: str = ""        # Google Gemini - free tier at aistudio.google.com
    qwen_api_key: str = ""          # Alibaba Qwen - free tier at dashscope.aliyuncs.com

    # GitHub
    github_token: str = ""
    github_default_org: str = ""

    # PostgreSQL
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_swe_team"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ai_swe_team"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Docker sandbox
    sandbox_image: str = "ai-swe-sandbox:latest"
    sandbox_network_disabled: bool = True

    # Git repository path for the Git MCP server
    repo_path: str = "."

    # LangSmith tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""

    # Graph control defaults
    max_revisions: int = 3


settings = Settings()
