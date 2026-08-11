import uuid
import logging

def generate_request_id() -> str:
    return str(uuid.uuid4())

def attach_request_id(logger: logging.Logger, request_id: str) -> logging.LoggerAdapter:
    class RequestIdAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f"request_id={self.extra['request_id']} {msg}", kwargs

    return RequestIdAdapter(logger, {"request_id": request_id})
