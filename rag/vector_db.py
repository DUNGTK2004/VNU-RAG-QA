import os
from embeding import embedding_model          
from document_loader import Loader           
from langchain.vectorstores import Chroma    
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.schema import Document      

vectordb_dir = "vectordb"  # Thư mục chính để lưu các vectorstore
if not os.path.exists(vectordb_dir):
    os.makedirs(vectordb_dir)  # Tạo thư mục nếu chưa tồn tại

class VectorDB:
    def __init__(
        self, 
        embeding_model=None, 
        vector_store=Chroma, 
        documents: list[Document] = None
    ):
        # Nếu không truyền mô hình embedding thì dùng mặc định
        self.embeding_model = embeding_model or embedding_model(model_name="BAAI/bge-m3")
        self.vector_store = vector_store
        self.documents = documents
        
        # Khởi tạo hoặc load vectorstore từ thư mục
        self.vectordb = self._create_vector_store(documents, "chroma_data")

    def _create_vector_store(self, docs: list[Document], store_name: str):
        """Tạo mới hoặc load vector store đã lưu"""
        persist_directory = os.path.join(vectordb_dir, store_name)

        if not os.path.exists(persist_directory):
            print(f"Creating new vector store at {persist_directory}")
            vectordb = self.vector_store.from_documents(
                documents=docs,
                embedding=self.embeding_model,
                persist_directory=persist_directory
            )
            vectordb.persist()
        else:
            print(f"Loading existing vector store from {persist_directory}")
            vectordb = self.vector_store(
                persist_directory=persist_directory,
                embedding_function=self.embeding_model
            )

        return vectordb

    def add_new_documents_to_store(self, new_docs: list[Document], store_name: str):
        persist_directory = os.path.join(vectordb_dir, store_name)
        if not os.path.exists(persist_directory):

            print(f"Vector store không tồn tại tại {persist_directory}. Không thể thêm tài liệu mới.")
            return self._create_vector_store(new_docs, store_name)
        else:
            print(f"Loading existing vector store from {persist_directory}")
            vectordb = self.vector_store(
                persist_directory=persist_directory,
                embedding_function=self.embeding_model
            )
            vectordb.add_documents(new_docs)
            vectordb.persist()
            return vectordb

    def get_retriever(
        self, 
        search_kwargs: dict = {"k": 10}, 
        search_type: str = "similarity"
    ):
        """Tạo retriever đơn giản từ vectorstore"""
        return self.vectordb.as_retriever(
            search_type=search_type, 
            search_kwargs=search_kwargs
        )

    def get_retriever_with_reranker(
        self, 
        search_kwargs: dict = {"k": 5}, 
        search_type: str = "similarity", 
        reranker_model_name: str = "BAAI/bge-reranker-base", 
        top_n: int = 5
    ):
        """Tạo retriever có thêm bước rerank kết quả bằng mô hình cross-encoder"""
        retriever = self.get_retriever(
            search_kwargs=search_kwargs, 
            search_type=search_type
        )

        # Dùng cross encoder model để rerank
        model = HuggingFaceCrossEncoder(model_name=reranker_model_name)

        # Cấu hình compressor chọn top_n kết quả tốt nhất sau rerank
        compressor = CrossEncoderReranker(
            model=model,
            top_n=top_n
        )


        # Kết hợp compressor và retriever để tạo ContextualCompressionRetriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

        return compression_retriever
    
    
    def get_documents_after_rerank(self, docs: list[Document], question: str, top_n: int = 5):

        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

        # Cấu hình compressor chọn top_n kết quả tốt nhất sau rerank
        compressor = CrossEncoderReranker(
            model=model,
            top_n=top_n
        )

        compressed_docs = compressor.compress_documents(docs, question)

        return compressed_docs

    def combine_documents(self, docs: list[Document]):
        """Kết hợp các tài liệu thành một chuỗi văn bản"""
        combined_text = ""
        for doc in docs:
            combined_text += doc.page_content + "\n"
        return combined_text


if __name__ == "__main__":
    # Load tài liệu từ thư mục test_data (dạng .txt)
    loader = Loader(json_path='test_data/cleaned_data.json')
    documents = loader.load_json()
    print(f"Loaded {len(documents)} documents.")
    # In ra nội dung của document đầu tiên sau khi split
    if documents:
        print(documents[0].page_content)
    else:
        print("No documents found.")
    # Khởi tạo vector DB và retriever có reranker
    vector_db = VectorDB(documents=documents)
    # update_vectordb = vector_db.add_new_documents_to_store(documents, "chroma_data")
    retriever = vector_db.get_retriever(search_kwargs={"k": 10})
    
    # Truy vấn văn bản
    query = " đại học quốc gia hà nội có địa chỉ là?"
    relevant_docs = retriever.get_relevant_documents(query)

    # In ra các tài liệu phù hợp
    print(f"Found {len(relevant_docs)} relevant documents.")
    for doc in relevant_docs:
        print("---------------------------------------------")
        print(doc.page_content)
