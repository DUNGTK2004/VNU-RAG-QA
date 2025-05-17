import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
from langchain_core.documents import Document



class FolderLoader: 
    """
    Load tất cả file trong thư mục theo pattern glob (ví dụ: *.txt).
    """
    def __init__(self, folder_path: str = 'test_data', glob: str = '*.txt'):
        self.folder_path = folder_path
        self.glob = glob

    def load(self) -> list[Document]:
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Directory '{self.folder_path}' does not exist.")
        
        loader = DirectoryLoader(self.folder_path, glob=self.glob)
        return loader.load()
    
    def load_json(self, file_path: str) -> list[Document]:
        """
        Load file JSON và chuyển đổi thành danh sách Document.
        """
        data = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Bây giờ biến 'data' chứa nội dung của file JSON dưới dạng cấu trúc dữ liệu Python
            # (thường là dictionary hoặc list)

            print("Đã load dữ liệu JSON thành công:")
            print(type(data)) # In ra kiểu dữ liệu (list hoặc dict)
            # print(data) # Có thể in toàn bộ dữ liệu để kiểm tra

        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}")
        except json.JSONDecodeError:
            print(f"Lỗi: Không thể giải mã file JSON {file_path}. Có thể file bị lỗi định dạng.")
        except Exception as e:
            print(f"Đã xảy ra lỗi khác: {e}")

        # 1. Trích xuất text và metadata
        documents = []
        for chunk_data in data:

            # Tạo dictionary metadata, loại bỏ key 'text'
            metadata = chunk_data.copy() 
            # del metadata["text"] # Tùy chọn: có thể giữ lại text trong metadata nếu muốn
            if "text" in metadata: # Nên kiểm tra key tồn tại trước khi xóa để tránh lỗi KeyError
                del metadata["text"]

            document = Document(
                page_content=chunk_data["text"],
                metadata=metadata
            )

            documents.append(document)

        return documents
class TextSplitter:
    """
    Chia nhỏ tài liệu thành các đoạn text có độ dài cố định với độ chồng lấn.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            # separators=["\n\n", "\n", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_text(self, documents: list[Document]) -> list[Document]:
        return self.splitter.split_documents(documents)

class Loader:
    """
    Class tổng hợp để load và split tài liệu từ thư mục.
    """
    def __init__(self, folder_path: str = 'test_data', glob: str = '*.txt', json_path: str = None):
        self.folder_loader = FolderLoader(folder_path, glob)
        self.text_splitter = TextSplitter()
        self.json_path = json_path
    def load(self) -> list[Document]:
        documents = self.folder_loader.load()
        return self.text_splitter.split_text(documents)

    def load_json(self) -> list[Document]:
        if self.json_path is None:
            raise ValueError("json_path must be provided to load JSON data.")
        documents = self.folder_loader.load_json(self.json_path)
        return self.text_splitter.split_text(documents)

if __name__ == "__main__":
    # load = FolderLoader(folder_path='test_data', glob='*.txt')
    # documents = load.load_json()
    # print(type(documents[0]))
    # Khởi tạo loader và load tài liệu đã được split
    loader = Loader(folder_path='test_data', glob='*.txt', json_path='test_data/all_chunks.json')
    documents = loader.load_json()

    # In ra nội dung của document đầu tiên sau khi split
    if documents:
        print(documents)
    else:
        print("No documents found.")
    


    # file_path = r'C:\Users\dungtk\Documents\VNU-RAG-QA\rag\test_data\all_chunks.json' # Thay thế bằng đường dẫn đến file JSON của bạn

    # try:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         data = json.load(f)

    #     # Bây giờ biến 'data' chứa nội dung của file JSON dưới dạng cấu trúc dữ liệu Python
    #     # (thường là dictionary hoặc list)

    #     print("Đã load dữ liệu JSON thành công:")
    #     print(type(data)) # In ra kiểu dữ liệu (list hoặc dict)
    #     # print(data) # Có thể in toàn bộ dữ liệu để kiểm tra

    # except FileNotFoundError:
    #     print(f"Lỗi: Không tìm thấy file tại đường dẫn {file_path}")
    # except json.JSONDecodeError:
    #     print(f"Lỗi: Không thể giải mã file JSON {file_path}. Có thể file bị lỗi định dạng.")
    # except Exception as e:
    #     print(f"Đã xảy ra lỗi khác: {e}")

    # # 1. Trích xuất text và metadata

    # documents = []
    # for chunk_data in data:

    #     # Tạo dictionary metadata, loại bỏ key 'text'
    #     metadata = chunk_data.copy() 
    #     # del metadata["text"] # Tùy chọn: có thể giữ lại text trong metadata nếu muốn
    #     if "text" in metadata: # Nên kiểm tra key tồn tại trước khi xóa để tránh lỗi KeyError
    #         del metadata["text"]

    #     document = Document(
    #         page_content=chunk_data["text"],
    #         metadata=metadata
    #     )

    #     documents.append(document)
    # # print(documents)
    


