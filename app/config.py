from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: str | None = None
    LLM_MODEL: str = "llama3-8b-8192"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    
    LANCEDB_PATH: str = "data/processed/lancedb"
    VECTOR_BACKEND: str = "lancedb"
    
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    
    DEFAULT_TOP_K: int = 3
    MIN_TOP_K: int = 1
    MAX_TOP_K: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

def is_groq_configured() -> bool:
    return bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip())

if not is_groq_configured():
    from app.telemetry import logger
    logger.warning("WARNING: GROQ_API_KEY is not configured or is invalid. Generation and LLM-as-judge features will fail or be skipped.")
