from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

GEN_MODEL_NAME = "google/flan-t5-base"

gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
gen_model.eval()


def build_prompt(context_chunks, question):
    """
    Construct a grounded prompt for the LLM.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""
    Answer the question using ONLY the context below.
    If the answer is not present in the context, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt.strip()


def generate_answer(context_chunks, question, max_tokens=256):
    """
    Generate an answer from retrieved chunks and a user query.
    """
    prompt = build_prompt(context_chunks, question)

    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=max_tokens
        )

    answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def run_rag(query, index, chunks, retrieve_fn, top_k=5):
    """
    End-to-end RAG execution.
    
    Args:
        query (str): User question
        index: FAISS index
        chunks (list): Text chunks
        retrieve_fn (callable): Retrieval function
        top_k (int): Number of chunks to retrieve
    """
    retrieved_chunks = retrieve_fn(query, index, chunks, top_k=top_k)

    answer = generate_answer(
        context_chunks=retrieved_chunks,
        question=query
    )

    return {
        "question": query,
        "context": retrieved_chunks,
        "answer": answer
    }


if __name__ == "__main__":
    # Import processing pipeline functions
    from processing_pipeline import (
        load_pdf_context,
        chunk_text,
        build_vector_store,
        retrieve_chunks,
    )

    # Step 1: Prepare data
    text = load_pdf_context()
    chunks = chunk_text(text)
    index = build_vector_store(chunks)

    # Step 2: Ask a question
    # query = "What is the structure of the cell membrane?"
    query = "What is a molecule?"

    result = run_rag(
        query=query,
        index=index,
        chunks=chunks,
        retrieve_fn=retrieve_chunks,
        top_k=5,
    )

    # Step 3: Output
    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])
