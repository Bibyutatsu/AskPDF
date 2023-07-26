import streamlit as st
from openai.error import OpenAIError

from utils import (
    create_sidebar, 
    wrap_text_in_html, 
    is_valid
)

from process import (
    embed_docs,
    get_answer,
    get_sources,
    parse_pdf,
    text_to_docs,
    get_images,
    image_grids,
    get_source_dict,
    get_source_images
)

def clear_submit():
    st.session_state["submit"] = False

if "expanded" not in st.session_state:
    st.session_state["expanded"] = True

st.set_page_config(page_title="AskPDF", layout="wide")
st.header("AskPDF")

create_sidebar()
print("Sidebar created")

uploaded_file = st.file_uploader(
    "Upload a pdf file",
    type=["pdf"],
    help="Scanned documents are not supported yet!",
    on_change=clear_submit,
)
print("Uploaded file created")

index = None
texts = None
images = None
if uploaded_file is not None:
    pdf = uploaded_file.read()
    texts, images, text_blocks = parse_pdf(pdf)
    docs = text_to_docs(texts)
    print("All text images and text blocks calculated")

    try:
        with st.spinner("Indexing document... This may take a while⏳"):
            index = embed_docs(docs)
            print("Embedding")
    except OpenAIError as e:
        st.error(e._message)

query = st.text_area("Ask a question about the document", on_change=clear_submit)

with st.expander("Advanced Options"):
    show_all_chunks = st.checkbox("Show all chunks retrieved from vector search")
    show_full_doc = st.checkbox("Show parsed contents of the document")
    show_images = st.checkbox("Show pdf pages of the document")
    show_sources = st.checkbox("Show pdf pages of the sources in answer")

if show_full_doc and texts:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_text_in_html(texts)}</p>", unsafe_allow_html=True)

if show_images and images:
    with st.expander("Images", st.session_state["expanded"]):
        number_of_columns = st.number_input("select grid width", 1,5,3, key="1st")
        image_grids(images, number_of_columns)
        if st.session_state["expanded"]:
            unexpand = st.button("Collapse")
            if unexpand:
                st.session_state["expanded"] = False
                st.experimental_rerun()
        else:
            expand = st.button("Expand")
            if expand:
                st.session_state["expanded"] = True
                st.experimental_rerun()

button = st.button("Submit")
if button or st.session_state.get("submit"):
    if not is_valid(index, query):
        st.stop()

    st.session_state["submit"] = True
    sources = index.similarity_search(query, k=5)  # type: ignore
    print("Sources calculated")
    try:
        answer = get_answer(sources, query)
        print("Answer calculated")
        
        if not show_all_chunks:
            sources = get_sources(answer, sources)

        st.markdown("#### Answer")
        st.markdown(answer["output_text"].split("SOURCES: ")[0])

        st.markdown("#### Sources")
        source_dict = get_source_dict(sources, text_blocks)
        print("Source Dict calc")
        
        if show_sources and sources:
            image_source = get_source_images(source_dict, images)
            number_of_columns_sources = st.number_input("select grid width", 1,5,3, key="2nd")
            image_grids(image_source, number_of_columns_sources)
            print("Image grids")
            
        else:
            for page in sorted(source_dict.keys()):
                st.markdown("Page " + str(page))
                source_per_page = sorted(source_dict[page], key=lambda x: int(x[1].split('-')[-1]))
                for content, source_index, text_block in source_per_page:
                    st.markdown(content)
                    st.markdown(source_index)
                st.markdown("---")

    except OpenAIError as e:
        st.error(e._message)
