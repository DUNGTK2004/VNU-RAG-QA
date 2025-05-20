from llm_model import get_llm, get_api_llm
from vector_db import VectorDB
from langchain_core.prompts import PromptTemplate
import time
import requests
# from config import *

from evaluation import compute_metric_general

def combine_context(context):
    combined_context = ""
    for doc in context:
        combined_context += doc.page_content + "\n"
    return combined_context

vector_db = VectorDB()
retriever = vector_db.get_retriever(search_kwargs={"k": 5})

# template = """Bạn là một trợ lý cho các tác vụ hỏi-đáp. Sử dụng các đoạn ngữ cảnh được truy xuất sau đây để trả lời câu hỏi. Không trả lời quá một dòng. Đừng dài dòng khi đưa ra câu trả lời. Đưa ra câu trả lời trực tiếp ngay cả khi nó không tạo thành một câu hoàn chỉnh.
# Câu hỏi:
# * {query} *
# Ngữ cảnh được cung cấp:
# * {context} *
# Trả lời:"""

template = """Dựa trên ngữ cảnh, chỉ cung cấp câu trả lời 1 lần 1 cách trực tiếp chỉ nằm trên 1 dòng dù không hoàn chỉnh, ngắn gọn nhất cho câu hỏi. 
            Không thêm từ ngữ thừa, giới thiệu hay giải thích. Trả lời bằng tiếng Việt.
            Ví dụ 1:
            Câu hỏi: Viện Trần Nhân Tông được thành lập năm nào?
            Câu trả lời: Năm 2016.

            Ví dụ 2:
            Câu hỏi: Trung tâm Hỗ trợ Sinh viên của Đại học Quốc gia Hà Nội trước đây có tên là gì?
            Câu trả lời: Trung tâm Nội trú Sinh viên.

            Đến lượt bạn:
            ngữ cảnh: {context}
            Câu hỏi: {query}
            Câu trả lời:
            """

# template_question = """
#     Tạo 3 câu hỏi biến thể từ câu hỏi dưới (giữ nguyên nội dung, chỉ diễn đạt khác):
#     Chỉ xuất đúng 3 câu hỏi, không kèm lời giới thiệu hay giải thích.
#     Câu hỏi gốc: "{query}"
# """

template_question = """
    [NHIỆM VỤ]: Tạo ra 3 dạng khác nhau của câu hỏi dưới đây, giữ nguyên nội dung. Không kèm lời giới thiệu.
    [CÂU HỎI]: {query}
"""



prompt = PromptTemplate(
    input_variables=["query", "context"],
    template=template
)

prompt_question = PromptTemplate(
    input_variables=[ "query"],
    template=template_question
)

# LLM API
llm = get_api_llm()

def remove_duplicate_retrieve_documents(documents):
    unique_documents = [doc for i, doc in enumerate(documents) if doc not in documents[:i]]
    return unique_documents


# --------------------------- Retriever --------------------------- ###
# --- File Processing ---
input_filename = r"test_data/question_all.txt"  # Tên file chứa các câu hỏi (mỗi câu 1 dòng)
output_filename = r"test_data/answers_retriever.txt"    # Tên file sẽ ghi các câu trả lời (mỗi câu 1 dòng)

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
                t1 = time.time()
                # --- RAG Process for the current query ---
                # Get relevant documents from the retriever
                context = retriever.get_relevant_documents(query)
                context = remove_duplicate_retrieve_documents(context)

                # Combine the context documents
                combined_context = vector_db.combine_documents(context)
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
                print("Answer: ", response.content)
                t2 = time.time()
                print("Time taken: ", t2 - t1)
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


with open(output_filename, 'r', encoding='utf-8') as outfile:
    answers = outfile.readlines()
    predicts = [answer.strip() for answer in answers]

expect_file = r"test_data/answer_all.txt"
with open(expect_file, 'r', encoding='utf-8') as expect_outfile:
    expect_answers = expect_outfile.readlines()
    expects = [answer.strip() for answer in expect_answers]

# Compute metrics
results = compute_metric_general(predicts, expects)


print('-------------results: ------------------')
print(results)

# ----------- Multiquery Test -----------

def split_questions(string, delimiter='?'):
    questions = string.split(delimiter)
    questions = [question.strip().split(". ", 1)[-1] + delimiter for question in questions]
    return questions


# query = "Tên tiếng Anh của Đại học Quốc gia Hà Nội là gì?"
# # Format the prompt with context and query
# formatted_prompt = prompt_question.format(query=query)

# # Invoke the LLM to get the response
# response = llm.invoke(formatted_prompt)

# print("Response: ", response.content)

# questions = split_questions(response.content)[:-1]
# print("Questions: ", questions) 
# context_arr = []
# for question in questions:
#     context_arr.extend(retriever.get_relevant_documents(question, ))

# docs_after_rerank = vector_db.get_documents_after_rerank(context_arr, question, top_n=10)
# combined_context = vector_db.combine_documents(docs_after_rerank)

# formatted_prompt = prompt.format(context=combined_context, query=query)
# print("Formatted Prompt: ", formatted_prompt)
# response = llm.invoke(formatted_prompt)
# print(response.content)



# ----------------------- Multiquery Test with many questions ------------------------

# --- File Processing --- 

input_filename = r"test_data/question_all.txt"  # Questions file (one per line)
output_filename = r"test_data/answers_multi-query.txt"   # Answers file (one per line)

print(f"Reading questions from '{input_filename}' and writing answers to '{output_filename}'...")

try:
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile):
            query = line.strip()
            
            # Skip empty lines
            if not query:
                print(f"Line {line_num + 1}: Empty line, skipping.")
                outfile.write("\n")  # Maintain correspondence
                continue
            
            print(f"Processing Line {line_num + 1}: '{query}'")
            
            try:
                t1 = time.time()
                # Format prompt and get initial response
                formatted_prompt = prompt_question.format(query=query)
                response = llm.invoke(formatted_prompt)
                
                # RAG Process
                context = retriever.get_relevant_documents(query)
                questions = split_questions(response.content)[:-1]
                print("Questions:", questions)
                
                # Get and rerank documents
                context_arr = context 
                for question in questions:
                    context_tmp = retriever.get_relevant_documents(question)
                    context_arr.extend(context_tmp)

                context_arr = remove_duplicate_retrieve_documents(context_arr)

                docs_after_rerank = vector_db.get_documents_after_rerank(context_arr, question, top_n=10)
                combined_context = vector_db.combine_documents(docs_after_rerank)
                
                # Generate final response
                formatted_prompt = prompt.format(context=combined_context, query=query)
                response = llm.invoke(formatted_prompt)
                t2 = time.time()
                print("Time taken: ", t2 - t1)
                # Write answer to output file
                print("Answer: ", response.content)
                outfile.write(response.content + "\n")
                print(f" --> Answer written for Line {line_num + 1}")
                
            except Exception as e:
                print(f"Error processing Line {line_num + 1} ('{query}'): {e}")
                outfile.write(f"Error processing question: {e}\n")

except FileNotFoundError:
    print(f"Error: Input file '{input_filename}' not found. Please create this file with your questions.")
except Exception as e:
    print(f"An unexpected error occurred during file processing: {e}")

print("Processing complete.")

with open(output_filename, 'r', encoding='utf-8') as outfile:
    answers = outfile.readlines()
    predicts = [answer.strip() for answer in answers]

expect_file = r"test_data/answer_all.txt"
with open(expect_file, 'r', encoding='utf-8') as expect_outfile:
    expect_answers = expect_outfile.readlines()
    expects = [answer.strip() for answer in expect_answers]

# Compute metrics
print(len(predicts), len(expects))
results = compute_metric_general(predicts, expects)


print('-------------results: ------------------')
print(results)

# Câu hỏi: Đại học Quốc gia Hà Nội có địa chỉ ở đâu?
# Câu trả lời: 144 Xuân Thủy, Cầu Giấy, Hà Nội.

# Câu hỏi: Ai là người viết tác phẩm Truyện Kiều?
# Câu trả lời: Câu hỏi không liên quan.

# Câu hỏi: Mùa đông ở Hà Nội thường vào tháng mấy?
# Câu trả lời: Câu hỏi không liên quan.

# Câu hỏi: Trường Đại học Bách khoa Hà Nội được thành lập năm nào?
# Câu trả lời: 1956.


## --------------------------- Retriever and Reranker --------------------------- ###
# --- File Processing ---

retriever = vector_db.get_retriever_with_reranker(search_kwargs={"k": 10}, top_n=5)

input_filename = r"test_data/question_all.txt"  # Tên file chứa các câu hỏi (mỗi câu 1 dòng)
output_filename = r"test_data/answers_rerank.txt"    # Tên file sẽ ghi các câu trả lời (mỗi câu 1 dòng)

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
                t1 = time.time()
                # --- RAG Process for the current query ---
                # Get relevant documents from the retriever
                context = retriever.get_relevant_documents(query)
                context = remove_duplicate_retrieve_documents(context)

                # Combine the context documents
                combined_context = vector_db.combine_documents(context)
                # print("--- Retrieved Context ---") # Optional: print context for debugging
                # print(combined_context)
                # print("-------------------------")

                # Format the prompt with context and query
                formatted_prompt = prompt.format(context=combined_context, query=query)

                # Invoke the LLM to get the response
                response = llm.invoke(formatted_prompt)
                print("Answer: ", response.content)
                # print("--- LLM Response ---") # Optional: print response for debugging
                # print(response)
                # print("--------------------")
                t2 = time.time()
                print("Time taken: ", t2 - t1)
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


## --------------------------- Non RAG --------------------------- ###
# # --- File Processing ---
# input_filename = r"test_data/question_all.txt"  # Tên file chứa các câu hỏi (mỗi câu 1 dòng)
# output_filename = r"test_data/answers.txt"    # Tên file sẽ ghi các câu trả lời (mỗi câu 1 dòng)

# print(f"Reading questions from '{input_filename}' and writing answers to '{output_filename}'...")

# # Open input and output files using 'with' for automatic closing
# try:
#     with open(input_filename, 'r', encoding='utf-8') as infile, \
#          open(output_filename, 'w', encoding='utf-8') as outfile:

#         # Process each line (question) in the input file
#         for line_num, line in enumerate(infile):
#             query = line.strip()  # Remove leading/trailing whitespace and newline characters

#             # Skip empty lines
#             if not query:
#                 print(f"Line {line_num + 1}: Empty line, skipping.")
#                 outfile.write("\n") # Write an empty line in output to maintain correspondence
#                 continue

#             print(f"Processing Line {line_num + 1}: '{query}'")

#             try:
#                 t1 = time.time()
#                 # --- RAG Process for the current query ---
#                 # Get relevant documents from the retriever
#                 # context = retriever.get_relevant_documents(query)
#                 # context = remove_duplicate_retrieve_documents(context)

#                 # # Combine the context documents
#                 # combined_context = vector_db.combine_documents(context)
#                 # print("--- Retrieved Context ---") # Optional: print context for debugging
#                 # print(combined_context)
#                 # print("-------------------------")

#                 # Format the prompt with context and query
#                 # formatted_prompt = prompt.format(context=combined_context, query=query)
#                 formatted_prompt = prompt.format( query=query)
#                 # Invoke the LLM to get the response
#                 response = llm.invoke(formatted_prompt)
#                 # print("--- LLM Response ---") # Optional: print response for debugging
#                 # print(response)
#                 # print("--------------------")
#                 print("Answer: ", response.content)
#                 t2 = time.time()
#                 print("Time taken: ", t2 - t1)
#                 # --- Write the response to the output file ---
#                 outfile.write(response.content + "\n") # Write the response followed by a newline

#                 print(f" --> Answer written for Line {line_num + 1}")

#             except Exception as e:
#                 # Handle errors during the processing of a single question
#                 print(f"Error processing Line {line_num + 1} ('{query}'): {e}")
#                 # Write an error message to the output file to indicate failure for this line
#                 outfile.write(f"Error processing question: {e}\n")

# except FileNotFoundError:
#     print(f"Error: Input file '{input_filename}' not found. Please create this file with your questions.")
# except Exception as e:
#     print(f"An unexpected error occurred during file processing: {e}")

# print("Processing complete.")


# with open(output_filename, 'r', encoding='utf-8') as outfile:
#     answers = outfile.readlines()
#     predicts = [answer.strip() for answer in answers]

# expect_file = r"test_data/answer_all.txt"
# with open(expect_file, 'r', encoding='utf-8') as expect_outfile:
#     expect_answers = expect_outfile.readlines()
#     expects = [answer.strip() for answer in expect_answers]

# # Compute metrics
# results = compute_metric_general(predicts, expects)


# print('-------------results: ------------------')
# print(results)


# # ----------------------- Test Together API -----------------------

# # url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"
# # token = HUGGINGFACE_API_KEY # Replace with your Hugging Face token


# # def llm(query, context):
# #    parameters = {
# #       "max_new_tokens": 50,
# #       "temperature": 0.01,
# #       "top_k": 50,
# #       "top_p": 0.95,
# #       "return_full_text": False
# #       }
  
# #    prompt = """<|begin_of_text|>
# #    <|start_header_id|>system<|end_header_id|>
# #    Bạn là trợ lý AI tiếng Việt. Trả lời chính xác, ngắn gọn nhất dựa trên ngữ cảnh {context}. Nếu không liên quan, trả lời "Câu hỏi không liên quan".
# #    <|eot_id|>

# #    <|start_header_id|>user<|end_header_id|>
# #    Ngữ cảnh: UET thành lập năm 1996.
# #    Câu hỏi: ```UET thành lập năm nào?```
# #    <|eot_id|>
# #    <|start_header_id|>assistant<|end_header_id|>
# #    1996.<|eot_id|>

# #    <|start_header_id|>user<|end_header_id|>
# #    Ngữ cảnh: Lịch thi ngày 15/12.
# #    Câu hỏi: ```Thủ đô Pháp?```
# #    <|eot_id|>
# #    <|start_header_id|>assistant<|end_header_id|>
# #    Câu hỏi không liên quan.<|eot_id|>

# #    <|start_header_id|>user<|end_header_id|>
# #    Ngữ cảnh: {context}
# #    Câu hỏi: ```{query}```
# #    <|eot_id|>
# #    <|start_header_id|>assistant<|end_header_id|>
# #    """

# #    headers = {
# #       'Authorization': f'Bearer {token}',
# #       'Content-Type': 'application/json'
# #    }
  
# #    prompt = prompt.replace("{query}", query)
# #    prompt = prompt.replace("{context}", context)
  
# #    payload = {
# #       "inputs": prompt,
# #       "parameters": parameters
# #    }
  
# #    response = requests.post(url, headers=headers, json=payload)
# #    response_text = response.json()[0]['generated_text'].strip()

# #    return response_text



# # vector_db = VectorDB()
# # retriever = vector_db.get_retriever_with_reranker(search_kwargs={"k": 5}, top_n=5)
# # query = "Đại học Quốc gia Hà Nội có bao nhiêu trường đại học thành viên?"
# # context = retriever.get_relevant_documents(query)
# # context = combine_context(context)

# # # print("Context: ", context)
# # response = llm(query, context)
# # print(response)