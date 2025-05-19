import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_core.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from rag.document_loader import Loader
from rag.vector_db import VectorDB
from langchain.chains import RetrievalQA


def init_llm(model_name: str = "meta-llama/Llama-2-7b-chat-hf", offload_dir: str = "./offload_folder") -> HuggingFacePipeline:
    """
    Initialize a local HuggingFace LLM with transformers and wrap it in a LangChain LLM.
    """
    # Ensure HF token is set via environment
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN not set in environment.")
    os.makedirs(offload_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        offload_folder=offload_dir,
    )
    text_gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=text_gen)


def main():
    # 1. Load and split JSON-based documents
    loader = Loader(json_path="data/all_chunks.json")
    documents = loader.load_json()

    # 2. Build or load vector store with embeddings
    vector_db = VectorDB(documents=documents)
    retriever = vector_db.get_retriever_with_reranker()

    # 3. Initialize LLM
    llm = init_llm()

    # 4. Define prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Dựa trên đoạn văn, chỉ cung cấp câu trả lời trực tiếp, ngắn gọn nhất cho câu hỏi. Không thêm từ ngữ thừa, giới thiệu hay giải thích. Trả lời bằng tiếng Việt. Nếu thông tin không có trong ngữ cảnh hoặc câu hỏi không liên quan, trả lời 'Câu hỏi không liên quan'.

Đoạn văn:
{context}

Câu hỏi:
{question}

Đáp án:"""
    )

    # 5. Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # 6. Read questions from file and save answers to reference_answers.txt
    questions_file = os.path.join("data", "test", "questions.txt")
    output_file = "data/test/reference_answers.txt"
    with open(questions_file, "r", encoding="utf-8") as qf:
        questions = [line.strip() for line in qf if line.strip()]
    answers = []
    for question in questions:
        answer = qa_chain.invoke(question)
        answers.append(answer)
    with open(output_file, "w", encoding="utf-8") as of:
        for ans in answers:
            of.write(ans.replace("\n", " ") + "\n")
    print(f"Saved {len(answers)} answers to {output_file}")


if __name__ == "__main__":
    main()