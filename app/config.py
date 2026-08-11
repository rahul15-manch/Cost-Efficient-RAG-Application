from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    LLM_MODEL: str = "llama3-8b-8192"
    EMBED_MODEL: str = "all-MiniLM-L6-v2"
    
    LANCEDB_PATH: str = "data/processed/lancedb"
    
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    
    DEFAULT_TOP_K: int = 3
    MIN_TOP_K: int = 1
    MAX_TOP_K: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
