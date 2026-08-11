import logging
import os
import time
from contextlib import contextmanager

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("trust-aware-rag")

@contextmanager
def Timer(operation_name: str):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000
    logger.info(f"{operation_name} completed in {elapsed_ms:.2f} ms")
