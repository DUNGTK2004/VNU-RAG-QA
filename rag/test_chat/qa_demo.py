import os
import torch
from model import QuestionAnsweringModel
from vector_db import VectorDB

def main():
    # Create a question answering model
    # Using XLM-RoBERTa which is good for multilingual tasks including Vietnamese

    # Create a vector database
    vectordb = VectorDB()
    retriever = vectordb.get_retriever_with_reranker(search_kwargs={"k": 5}, top_n=5)
    
    
    model = QuestionAnsweringModel(model_name="xlm-roberta-base")
    
    # # Load dataset
    # print("Loading dataset...")
    # train_dataset = model.load_dataset(split="train")
    
    # # Print dataset info
    # print(f"Dataset loaded with {len(train_dataset)} examples")
    # print(f"Sample example: {train_dataset[0]}")
    
    # # Preprocess the dataset
    # print("Preprocessing dataset...")
    # processed_train = model.preprocess_data(train_dataset)
    
    # # Create a small validation set (20% of train)
    # train_size = int(0.8 * len(processed_train))
    # val_size = len(processed_train) - train_size
    
    # # Split dataset
    # train_dataset_split = processed_train.select(range(train_size))
    # eval_dataset_split = processed_train.select(range(train_size, len(processed_train)))
    
    # print(f"Training with {len(train_dataset_split)} examples")
    # print(f"Evaluating with {len(eval_dataset_split)} examples")
    
    # Train the model (only if GPU is available or if explicitly requested)
    # if torch.cuda.is_available() or os.environ.get("FORCE_TRAIN", "0") == "1":
    #     print("Training model...")
    #     trainer = model.train(
    #         train_dataset=train_dataset_split,
    #         eval_dataset=eval_dataset_split,
    #         output_dir="./qa_model_vi",
    #         epochs=2,  # Reduce for faster training
    #         batch_size=4  # Adjust based on your GPU memory
    #     )
    #     print("Training complete!")
    # else:
    #     print("Skipping training as no GPU was detected. Set FORCE_TRAIN=1 to train on CPU.")
    
    # Example prediction
    # sample = train_dataset[0]
    question = "Dịch bệnh nào dẫn đến nhiều cái chết giữa những năm 1830 và 1860?"
    context = "Vô số người bản địa đã chiếm Alaska trong hàng ngàn năm trước khi các dân tộc châu Âu đến khu vực này. Các nghiên cứu ngôn ngữ và DNA được thực hiện ở đây đã cung cấp bằng chứng cho việc định cư Bắc Mỹ bằng cầu đất Bering. [Cần dẫn nguồn] Người Tlingit đã phát triển một xã hội với hệ thống thừa kế tài sản mẫu hệ và hậu duệ ở vùng Đông Nam Alaska ngày nay, cùng với các bộ phận của British Columbia và Yukon. Cũng ở Đông Nam là Haida, bây giờ nổi tiếng với nghệ thuật độc đáo của họ. Người Tsimshian đến Alaska từ British Columbia vào năm 1887, khi Tổng thống Grover Cleveland, và sau đó là Quốc hội Hoa Kỳ, cho phép họ định cư trên đảo Annette và tìm thấy thị trấn Metlakatla. Cả ba dân tộc này, cũng như các dân tộc bản địa khác của Bờ biển Tây Bắc Thái Bình Dương, đã trải qua bệnh đậu mùa từ cuối thế kỷ 18 đến giữa thế kỷ 19, với những dịch bệnh tàn khốc nhất xảy ra vào những năm 1830 và 1860 , dẫn đến tử vong cao và gián đoạn xã hội."
    expected_answer = "bệnh đậu mùa"
    
    print("\nExample Question Answering:")
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    
    # Get prediction
    prediction = model.predict(question, context)
    print(f"Predicted Answer: {prediction['text']}")

if __name__ == "__main__":
    main() 