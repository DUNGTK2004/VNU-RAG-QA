from evaluation import compute_metric 
import numpy as np
def test_statistic(true_label, out_method1, out_method2, num_boostrap=10000, bootstrap_ratio=1):
    """
    Hàm kiểm tra sự khác biệt giữa hai phương pháp dự đoán bằng cách sử dụng kiểm định thống kê.
    """
    # Tính toán các chỉ số đánh giá cho cả hai phương pháp
    f1_score_method1 = [compute_metric(true_label[i], out_method1[i]) for i in range(len(true_label))]
    f1_score_method2 = [compute_metric(true_label[i], out_method2[i]) for i in range(len(true_label))]

    sys1_win = 0
    sys2_win = 0
    sys1_tie = 0

    for i in range(num_boostrap):
        # Chọn ngẫu nhiên một tỷ lệ mẫu từ dữ liệu
        sample_indices = np.random.choice(len(true_label), int(len(true_label) * bootstrap_ratio), replace=True)
        sample_f1_method1 = [f1_score_method1[i] for i in sample_indices]
        sample_f1_method2 = [f1_score_method2[i] for i in sample_indices]
        # 
        score_f1_method1 = np.sum(sample_f1_method1) / len(sample_f1_method1)
        score_f1_method2 = np.sum(sample_f1_method2) / len(sample_f1_method2)
        # So sánh các chỉ số đánh giá
        if score_f1_method1 > score_f1_method2:
            sys1_win += 1
        elif score_f1_method1 < score_f1_method2:
            sys2_win += 1
        else:
            sys1_tie += 1
        
    ratio_sys1_win = sys1_win / float(num_boostrap)
    ratio_sys2_win = sys2_win / float(num_boostrap)
    ratio_sys1_tie = sys1_tie / float(num_boostrap)
    print(f"sys1_win: {sys1_win}, sys2_win: {sys2_win}, sys1_tie: {sys1_tie}")
    print(f"ratio_sys1_win: {ratio_sys1_win}, ratio_sys2_win: {ratio_sys2_win}, ratio_sys1_tie: {ratio_sys1_tie}")
   
    
    return {
        "sys1_win": sys1_win,
        "sys2_win": sys2_win,
        "sys1_tie": sys1_tie,
        "ratio_sys1_win": ratio_sys1_win,
        "ratio_sys2_win": ratio_sys2_win,
        "ratio_sys1_tie": ratio_sys1_tie
    }
if __name__ == "__main__":
    # Ví dụ sử dụng
    with open(r"test_data\answer_all.txt", "r", encoding="utf-8") as f:
        true_label = [line.strip() for line in f if line.strip()]
    with open("answers_retriever.txt", "r", encoding="utf-8") as f:
        out_method1 = [line.strip() for line in f if line.strip()]
    with open(r"answers_rerank.txt", "r", encoding="utf-8") as f:
        out_method2 = [line.strip() for line in f if line.strip()]

    result = test_statistic(true_label, out_method1, out_method2)
    print(result)