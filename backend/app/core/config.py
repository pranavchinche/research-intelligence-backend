from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Research Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DATABASE_URL: str = ""

    LOG_LEVEL: str = "INFO"

    UPLOAD_DIR: str = "uploads"

    ARXIV_BASE_URL: str = "http://export.arxiv.org/api/query"
    OPENALEX_BASE_URL: str = "https://api.openalex.org"

    PDF_STORAGE_DIR: str = "D:/FYP/main/storage/papers"
    PDF_DOWNLOAD_DIR: str = "D:/FYP/main/storage/downloads"

    CORS_ORIGINS: str = (
        "http://localhost:3001,"
        "http://192.168.137.1:3001,"
        "http://192.168.90.91:3001"
    )

    # LLM Provider Keys (all optional, environment-only)
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    # LLM Provider Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"
    OLLAMA_ENABLED: bool = True
    # OLLAMA_TIMEOUT: float = 0.0  # 0.0 => fall back to LLM_TIMEOUT

    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_ENABLED: bool = True
    # GROQ_TIMEOUT: float = 0.0

    GEMINI_BASE_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta"
    )
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_ENABLED: bool = True
    # GEMINI_TIMEOUT: float = 0.0

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"
    OPENROUTER_ENABLED: bool = True
    # OPENROUTER_TIMEOUT: float = 0.0

    LLM_DEFAULT_PROVIDER: str = "ollama"
    LLM_FALLBACK_CHAIN: str = "ollama"

    # LLM Generation Settings
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: float = 120.0

    # Redis
    REDIS_URL: str = ""

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: int = 60

    # Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 25
    MAX_PDF_PAGES: int = 200

    # Connectivity
    CONNECTIVITY_CHECK_INTERVAL: int = 30
    CONNECTIVITY_TIMEOUT: float = 5.0

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]

    @property
    def llm_fallback_chain(self) -> list[str]:
        return [
            p.strip()
            for p in self.LLM_FALLBACK_CHAIN.split(",")
            if p.strip()
        ]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def is_offline_mode(self) -> bool:
        return self.APP_ENV == "offline"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
