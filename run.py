import os
from askpdf.process import (
    embed_docs,
    get_answer,
    get_sources,
    parse_pdf,
    text_to_docs,
    get_images,
    get_source_dict,
    get_source_images
)

os.environ["NLPCLOUD_API_KEY"] = "<API Token here>"

pdf = input("Path to PDF file: ") ## Example: "./test/employment_agreement.pdf"
with open(pdf, 'rb') as pdffile:
    texts, images, text_blocks = parse_pdf(pdffile)

print("Reading PDF")
docs = text_to_docs(texts)

print("Calculating Embeddings for the content")
index = embed_docs(docs)

query = input("Enter the question you want to ask: ")  ## Example: "What is written in the document? Is it safe to be signed without reading?"
sources = index.similarity_search(query, k=5)
answer = get_answer(sources, query)
print(f"Answer: {answer['output_text']}")


