from llm_model import get_llm, get_llm_gemini, get_api_llm
from vector_db import VectorDB
from langchain_core.prompts import PromptTemplate

import requests
from config import *

def combine_context(context):
    combined_context = ""
    for doc in context:
        combined_context += doc.page_content + "\n"
    return combined_context

vector_db = VectorDB()
retriever = vector_db.get_retriever_with_reranker(search_kwargs={"k": 10}, top_n=10)

template = """Dựa trên {context}, chỉ cung cấp câu trả lời trực tiếp, ngắn gọn nhất cho câu hỏi. Không thêm từ ngữ thừa, giới thiệu hay giải thích. Trả lời bằng tiếng Việt. Nếu thông tin không có trong ngữ cảnh hoặc câu hỏi không liên quan, trả lời 'Câu hỏi không liên quan'.
Câu hỏi: {query}
Câu trả lời:"""

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)

# LLM API
llm = get_api_llm()


# --- File Processing ---
input_filename = r"test_data/questions.txt"  # Tên file chứa các câu hỏi (mỗi câu 1 dòng)
output_filename = r"test_data/answers.txt"    # Tên file sẽ ghi các câu trả lời (mỗi câu 1 dòng)

print(f"Reading questions from '{input_filename}' and writing answers to '{output_filename}'...")

# Open input and output files using 'with' for automatic closing
try:
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:

        # Process each line (question) in the input file
        for line_num, line in enumerate(infile):
            query = line.strip()  # Remove leading/trailing whitespace and newline characters

            # Skip empty lines
            if not query:
                print(f"Line {line_num + 1}: Empty line, skipping.")
                outfile.write("\n") # Write an empty line in output to maintain correspondence
                continue

            print(f"Processing Line {line_num + 1}: '{query}'")

            try:
                # --- RAG Process for the current query ---
                # Get relevant documents from the retriever
                context = retriever.get_relevant_documents(query)

                # Combine the context documents
                combined_context = combine_context(context)
                print(combined_context)
                # print("--- Retrieved Context ---") # Optional: print context for debugging
                # print(combined_context)
                # print("-------------------------")

                # Format the prompt with context and query
                formatted_prompt = prompt.format(context=combined_context, query=query)

                # Invoke the LLM to get the response
                response = llm.invoke(formatted_prompt)
                # print("--- LLM Response ---") # Optional: print response for debugging
                # print(response)
                # print("--------------------")

                # --- Write the response to the output file ---
                outfile.write(response.content + "\n") # Write the response followed by a newline

                print(f" --> Answer written for Line {line_num + 1}")

            except Exception as e:
                # Handle errors during the processing of a single question
                print(f"Error processing Line {line_num + 1} ('{query}'): {e}")
                # Write an error message to the output file to indicate failure for this line
                outfile.write(f"Error processing question: {e}\n")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found. Please create this file with your questions.")
except Exception as e:
    print(f"An unexpected error occurred during file processing: {e}")

print("Processing complete.")

# url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
# token = HUGGINGFACE_API_KEY # Replace with your Hugging Face token


# def llm(query, context):
#    parameters = {
#       "max_new_tokens": 50,
#       "temperature": 0.01,
#       "top_k": 50,
#       "top_p": 0.95,
#       "return_full_text": False
#       }
  
#    prompt = """<|begin_of_text|>
#    <|start_header_id|>system<|end_header_id|>
#    Bạn là trợ lý AI tiếng Việt. Trả lời chính xác, ngắn gọn nhất dựa trên ngữ cảnh {context}. Nếu không liên quan, trả lời "Câu hỏi không liên quan".
#    <|eot_id|>

#    <|start_header_id|>user<|end_header_id|>
#    Ngữ cảnh: UET thành lập năm 1996.
#    Câu hỏi: ```UET thành lập năm nào?```
#    <|eot_id|>
#    <|start_header_id|>assistant<|end_header_id|>
#    1996.<|eot_id|>

#    <|start_header_id|>user<|end_header_id|>
#    Ngữ cảnh: Lịch thi ngày 15/12.
#    Câu hỏi: ```Thủ đô Pháp?```
#    <|eot_id|>
#    <|start_header_id|>assistant<|end_header_id|>
#    Câu hỏi không liên quan.<|eot_id|>

#    <|start_header_id|>user<|end_header_id|>
#    Ngữ cảnh: {context}
#    Câu hỏi: ```{query}```
#    <|eot_id|>
#    <|start_header_id|>assistant<|end_header_id|>
#    """

#    headers = {
#       'Authorization': f'Bearer {token}',
#       'Content-Type': 'application/json'
#    }
  
#    prompt = prompt.replace("{query}", query)
#    prompt = prompt.replace("{context}", context)
  
#    payload = {
#       "inputs": prompt,
#       "parameters": parameters
#    }
  
#    response = requests.post(url, headers=headers, json=payload)
#    response_text = response.json()[0]['generated_text'].strip()

#    return response_text



# vector_db = VectorDB()
# retriever = vector_db.get_retriever_with_reranker(search_kwargs={"k": 5}, top_n=5)
# query = "Đại học Quốc gia Hà Nội có bao nhiêu trường đại học thành viên?"
# context = retriever.get_relevant_documents(query)
# context = combine_context(context)

# # print("Context: ", context)
# response = llm(query, context)
# print(response)