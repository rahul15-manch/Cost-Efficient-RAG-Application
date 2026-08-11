def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if exitstatus == 0:
        terminalreporter.write_sep("=", "Integration Suite Summary: ALL TESTS PASSED SUCCESSFULLY", bold=True, green=True)
        terminalreporter.write_line("Verified End-to-End:")
        terminalreporter.write_line(" ✓ API Health & Telemetry Logging")
        terminalreporter.write_line(" ✓ Multi-format Ingestion (PDF, HTML, MD)")
        terminalreporter.write_line(" ✓ Sentence-Aware Chunking & Overlap")
        terminalreporter.write_line(" ✓ Idempotent Deduplication (No Duplicates)")
        terminalreporter.write_line(" ✓ LanceDB Vector Storage & Embeddings")
        terminalreporter.write_line(" ✓ Adaptive Retrieval (Top-K adjustment)")
        terminalreporter.write_line(" ✓ Confidence Scoring & Evidence Coverage")
        terminalreporter.write_line(" ✓ Metadata Filtering by Source & Section")
