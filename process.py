import re
from io import BytesIO
from typing import Any, Dict, List
import docx2txt
from PIL import Image, ImageDraw
import streamlit as st
from hashlib import md5
import fitz
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai.error import AuthenticationError
from langchain.prompts import PromptTemplate
from thefuzz import fuzz

from template import STUFF_PROMPT


def hash_func(doc: Document) -> str:
    """Hash function for caching Documents"""
    return md5(doc.page_content.encode("utf-8")).hexdigest()


@st.cache_data()
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.cache_data()
def parse_pdf(file: BytesIO) -> List[str]:
    try:
        pdf = fitz.open("pdf", file)
    except:
        try:
            pdf = fitz.open("pdf", file.read())
        except:
            raise ValueError("Unable to read file")
    output = []
    images = []
    text_blocks = []
    for page in pdf:
        ### Read text
        text = page.get_text()#.encode("utf8")
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
        
        ### Read images
        pix = get_image(page.get_pixmap())
        images.append(pix)
        
        ### Read TextBlocks
        text_block = page.get_text("Blocks")
        text_blocks.append(text_block)
    return output, images, text_blocks


@st.cache_data()
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.cache_data()
def text_to_docs(text: str | List[str]) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


@st.cache_data()
def parse_file(file: BytesIO, name: str) -> str | List[str]:
    """Parses a file and returns a list of Documents."""
    if name.endswith(".pdf"):
        return parse_pdf(file)
    elif name.endswith(".docx"):
        return parse_docx(file)
    elif name.endswith(".txt"):
        return parse_txt(file)
    else:
        raise ValueError("File type not supported!")


@st.cache_data(show_spinner=False, hash_funcs={Document: hash_func})
def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    if not st.session_state.get("OPENAI_API_KEY"):
        raise AuthenticationError(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
    else:
        # Embed the chunks
        embeddings = OpenAIEmbeddings(
            openai_api_key=st.session_state.get("OPENAI_API_KEY"),
        )  # type: ignore

        index = FAISS.from_documents(docs, embeddings)

        return index


@st.cache_data(show_spinner=False, hash_funcs={Document: hash_func})
def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer
    chain = load_qa_with_sources_chain(
        ChatOpenAI(
            temperature=0, openai_api_key=st.session_state.get("OPENAI_API_KEY")
        ),  # type: ignore
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    answer = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    return answer


@st.cache_data(show_spinner=False, hash_funcs={Document: hash_func})
def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def get_image(pix_map):
    img = Image.frombytes("RGB", [pix_map.width, pix_map.height], pix_map.samples)
    return img

@st.cache_data()
def get_images(file):
    pdf = fitz.open("pdf", file)
    images = []
    for page in pdf:
        pix = get_image(page.get_pixmap())
        images.append(pix)
    return images

@st.cache_data()
def image_grids(images, number_of_columns):
    for j in range(len(images)//number_of_columns + 1):
        cols = st.columns(number_of_columns)
        for i, image in enumerate(images[j*number_of_columns:(j+1)*number_of_columns]):
            cols[i].image(image)
            
    return

def process_text(text):
    return text.lower().replace("\n","").replace(" ","")

@st.cache_data()
def get_matches(text_blocks, text):
    for text_block in text_blocks:
        # if process_text(text) in process_text(text_block[4]):
        if fuzz.partial_ratio(process_text(text), process_text(text_block[4]))>80:
            return text_block[:4]

    return None

def draw_rect(img, list_of_coords):
    img = img.copy()
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for coords in list_of_coords:
        draw.rectangle(coords, fill=(255, 255, 0, 100))
    img = Image.alpha_composite(img, overlay)
    return img

@st.cache_data(hash_funcs={Document: hash_func})
def get_source_dict(sources, text_blocks):
    source_dict = {}
    for source in sources:
        content = source.page_content
        source_index = source.metadata["source"]
        page_no = int(source_index.split('-')[0])
        text_block = get_matches(text_blocks[page_no-1], content)

        if page_no not in source_dict.keys():
            source_dict[page_no] = [(content, source_index, text_block)]
        else:
            source_dict[page_no].append((content, source_index, text_block))
    
    return source_dict

@st.cache_data()
def get_source_images(source_dict, images):
    image_source =[]
    for page in sorted(source_dict.keys()):
        text_blocks = [x[2][:4] for x in source_dict[page] if x[2] is not None]
        if text_blocks:
            img = draw_rect(images[int(page)-1], text_blocks)
            image_source.append(img)
            
    return image_source
