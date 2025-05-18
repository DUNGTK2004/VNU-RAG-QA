import re
from collections import Counter

def normalize_answer(answer: str) -> str:
    """
    Hàm chuẩn hóa câu trả lời bằng cách loại bỏ các ký tự không cần thiết.
    """
    # Chuyển đổi thành chữ thường
    answer = answer.lower()

    # Loại bỏ các từ the, a, an
    regex = re.compile(r'\b(the|a|an)\b')
    answer = re.sub(regex,'', answer)
    
    # Loại bỏ các ký tự không phải chữ cái và số
    answer = ''.join(char for char in answer if char.isalnum() or char.isspace())
    
    # Loại bỏ khoảng trắng thừa
    answer = ' '.join(answer.split())
    
    return answer 


def is_exact_match(predicted_answer: str, expected_answer: str) -> bool:
    """
    Hàm kiểm tra xem câu trả lời dự đoán có khớp chính xác với câu trả lời mong đợi hay không.
    """
    # Chuẩn hóa cả hai câu trả lời
    normalized_predicted = normalize_answer(predicted_answer)
    normalized_expected = normalize_answer(expected_answer)

    # So sánh hai câu trả lời đã chuẩn hóa
    return int(normalized_predicted == normalized_expected)

def return_list_tokens(answer: str) -> list:
    if not answer or not isinstance(answer, str):
        return []
    return normalize_answer(answer).split()

def compute_metric(predicted_answer: str, expected_answer: str) -> dict:
    
    list_token_predicted = return_list_tokens(predicted_answer)
    list_token_expected = return_list_tokens(expected_answer)

    pre_counter = Counter(list_token_predicted)
    expected_counter = Counter(list_token_expected)
    common_counter = pre_counter & expected_counter
    num_common = sum(common_counter.values())

    # Tính các chỉ số đánh giá
    if len(list_token_predicted) == 0 or len(list_token_expected) == 0:
        f1_score = 0.0
        precision = 0.0
        recall = 0.0
    else:
        precision = num_common / len(list_token_predicted)
        recall = num_common / len(list_token_expected)
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall)
    
    return (
        f1_score,
        precision,
        recall
    )
    # 



def compute_metric_general(predicted_answers, expected_answers) -> dict:
    """
    Hàm tính toán các chỉ số đánh giá cho câu trả lời dự đoán.
    """
    num_exact_matches = 0
    total_f1_score = 0.0
    total_precision = 0.0
    total_recall = 0.0

    for i in range(len(predicted_answers)):
        predicted_answer = predicted_answers[i]
        expected_answer = expected_answers[i]

        # Kiểm tra xem câu trả lời có khớp chính xác không
        exact_match = is_exact_match(predicted_answer, expected_answer)

        # Tính toán chỉ số đánh giá
        f1_score, precision, recall = compute_metric(predicted_answer, expected_answer)

        print(type(total_f1_score), type(f1_score))
        # Cộng dồn các chỉ số
        num_exact_matches += exact_match
        total_f1_score += f1_score
        total_precision += precision
        total_recall += recall

    # Tính toán các số đánh giá trung bình
    num_samples = len(predicted_answers)
    avg_f1_score = total_f1_score / num_samples
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    avg_exact_match = num_exact_matches / num_samples
    return {
        "exact_match": avg_exact_match,
        "f1_score": avg_f1_score,
        "precision": avg_precision,
        "recall": avg_recall
    }

if __name__ == "__main__":
    predicted_answers = [
        "The quick brown fox jumps over the lazy dog",
        "A tall man is standing next to a cow and a boy",
        "Paris is the capital of France"
    ]

    expected_answers = [
        "quick brown fox is man tall",
        "The tall man cow boy",
        "Paris is the capital of France"
    ]

    result = compute_metric_general(predicted_answers, expected_answers)
    print(result)