import tempfile
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Research Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    DATABASE_URL: str = ""
    LOCAL_DATABASE_URL: str = ""

    # DB_ENV selects which database to use:
    #   "local"     -> LOCAL_DATABASE_URL
    #   "production"-> DATABASE_URL (e.g. Neon)
    DB_ENV: str = "local"

    LOG_LEVEL: str = "INFO"

    UPLOAD_DIR: str = "uploads"

    ARXIV_BASE_URL: str = "http://export.arxiv.org/api/query"
    OPENALEX_BASE_URL: str = "https://api.openalex.org"

    PDF_STORAGE_DIR: str = ""
    PDF_DOWNLOAD_DIR: str = ""

    # Google Drive configuration
    GOOGLE_DRIVE_FOLDER_ID: str = ""
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REFRESH_TOKEN: str = ""

    CORS_ORIGINS: str = "http://localhost:3001"

    # LLM Provider Keys (all optional, environment-only)
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    # LLM Provider Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"
    OLLAMA_ENABLED: bool = True

    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_ENABLED: bool = True

    GEMINI_BASE_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta"
    )
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_ENABLED: bool = True

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "meta-llama/llama-3.1-8b-instruct:free"
    OPENROUTER_ENABLED: bool = True

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

    @property
    def active_database_url(self) -> str:
        """Select the active database URL based on DB_ENV.

        DB_ENV=production -> DATABASE_URL (e.g. Neon)
        DB_ENV=local      -> LOCAL_DATABASE_URL
        """
        if self.DB_ENV == "production":
            return self.DATABASE_URL
        return self.LOCAL_DATABASE_URL or self.DATABASE_URL

    @property
    def use_google_drive(self) -> bool:
        return bool(
            self.GOOGLE_DRIVE_FOLDER_ID
            and self.GOOGLE_CLIENT_ID
            and self.GOOGLE_CLIENT_SECRET
            and self.GOOGLE_REFRESH_TOKEN
        )

    @property
    def resolved_pdf_storage_dir(self) -> str:
        if self.PDF_STORAGE_DIR:
            return self.PDF_STORAGE_DIR
        return tempfile.mkdtemp(prefix="pdf_storage_")

    @property
    def resolved_pdf_download_dir(self) -> str:
        if self.PDF_DOWNLOAD_DIR:
            return self.PDF_DOWNLOAD_DIR
        return tempfile.mkdtemp(prefix="pdf_download_")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()

# Ensure storage dirs exist for local mode
if not settings.use_google_drive:
    import os
    os.makedirs(settings.resolved_pdf_storage_dir, exist_ok=True)
    os.makedirs(settings.resolved_pdf_download_dir, exist_ok=True)
    # Backfill empty defaults so downstream code works
    if not settings.PDF_STORAGE_DIR:
        settings.PDF_STORAGE_DIR = settings.resolved_pdf_storage_dir
    if not settings.PDF_DOWNLOAD_DIR:
        settings.PDF_DOWNLOAD_DIR = settings.resolved_pdf_download_dir
