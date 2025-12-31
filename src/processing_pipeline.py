import faiss
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def load_pdf_context(start_page=19, end_page=68):
    """Load and extract text from PDF chapters"""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    pdf_path = PROJECT_ROOT / "data" / "ConceptsofBiology-WEB.pdf"

    reader = PdfReader(pdf_path)
    text = ""

    for page_num in range(start_page - 1, end_page):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def embed_texts(texts):
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        model_output = model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype("float32")


def build_vector_store(chunks):
    embeddings = embed_texts(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def retrieve_chunks(query, index, chunks, top_k=5):
    query_embedding = embed_texts([query])
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]
