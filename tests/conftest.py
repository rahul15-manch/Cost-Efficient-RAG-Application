import os

# Set test environment overrides BEFORE app import
os.environ["LANCEDB_PATH"] = "tests/test_lancedb"
os.environ["CHUNK_SIZE"] = "600"
os.environ["CHUNK_OVERLAP"] = "100"
# Note: We rely on .env for GROQ_API_KEY so generation tests use real inference

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if exitstatus == 0:
        terminalreporter.write_sep("=", "Trust-Aware RAG Integration Suite: PASSED", bold=True, green=True)
        terminalreporter.write_line("All pipeline checks succeeded.")
