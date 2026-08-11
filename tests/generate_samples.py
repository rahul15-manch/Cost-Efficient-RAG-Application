import fitz

with open("tests/sample.md", "w", encoding="utf-8") as f:
    f.write("# Evaluation\n\nRecall@k measures retrieval quality.\n\n## Cost\n\nLanceDB reduces infrastructure cost.\n")

with open("tests/sample.html", "w", encoding="utf-8") as f:
    f.write("<html><body><h1>Introduction</h1><p>Welcome to RAG.</p><h2>Details</h2><p>Here are the details.</p></body></html>")
    
doc = fitz.open()
page1 = doc.new_page()
page1.insert_text((50, 50), "PDF Content Page 1")
page2 = doc.new_page()
page2.insert_text((50, 50), "PDF Content Page 2")
doc.save("tests/sample.pdf")
doc.close()
