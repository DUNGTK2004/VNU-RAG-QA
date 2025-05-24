# 📚 Hướng dẫn sử dụng

## 1. Cài đặt thư viện

Trước tiên, bạn cần cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install -r requirements.txt
```

## 2. Tạo và sử dụng cơ sở dữ liệu vector

### Bước 1: Tạo cơ sở dữ liệu

Chạy file `vector_db.py` để tạo database vector (quá trình này có thể mất khá nhiều thời gian).  
Bạn có thể chỉnh sửa đường dẫn thư mục chứa database nếu cần.

### Bước 2: Khởi tạo và truy vấn

Trong file làm việc của bạn, import class `VectorDB` từ `vector_db.py` và khởi tạo đối tượng như sau:

```python
from vector_db import VectorDB

vectordb = VectorDB()  # Khởi tạo vectordb

retriever = vectordb.get_retriever(search_kwargs={"k": 10})  # Khởi tạo retriever

query = "đại học quốc gia hà nội có địa chỉ là?"  # Câu hỏi
relevant_docs = retriever.get_relevant_documents(query)  # Các context thu được
```

### Bước 3: Sử dụng kết quả

Sau bước trên, bạn sẽ thu được các đoạn văn bản (documents) phù hợp với câu hỏi.  
Hãy đưa chúng cùng với câu hỏi vào mô hình LLM để sinh ra câu trả lời.

---

> ⚠️ **Lưu ý:**  
> Nếu gặp lỗi không import được `vector_db.py` do khác thư mục, bạn cần tự xử lý đường dẫn import nhé.
