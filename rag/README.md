# Cách sử dụng:


Đầu tiên cài các thư viện:

`pip install -r requirements.txt`

import class VectorDB từ file vector_db.py và khởi tạo trong file làm việc của bạn. 

Đầu tiên chạy file vector_db.py để tạo database (chạy khá lâu) (có thể sửa đường dẫn của folder chứa database nếu muốn)

Sau khi đã có database thì dùng code này trong file của bạn để lấy ra context cho câu hỏi.
Example: 
```
from vector_db import VectorDB

vectordb = VectorDB() # Khởi tạo vectordb

retriever = vector_db.get_retriever(search_kwargs={"k": 10}) # Khởi tạo retriever

query = "đại học quốc gia hà nội có địa chỉ là?" # Câu hỏi
relevant_docs = retriever.get_relevant_documents(query) # Các context thu được
```
Xong đoạn này thì đây chính là những documents thu được từ quá trình retriver, sau đó thì lấy đoạn này đưa vào llm kết hợp với câu hỏi để sinh ra câu trả lời .

Thường thì sẽ bị lỗi là ở thư mục khác nên ko import được từ thư mục này, cái đấy thì tự xử lý đi

