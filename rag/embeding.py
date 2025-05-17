from langchain_huggingface import HuggingFaceEmbeddings

def embedding_model(model_name: str = "dangvantuan/vietnamese-embedding"):
    return HuggingFaceEmbeddings(model_name=model_name)