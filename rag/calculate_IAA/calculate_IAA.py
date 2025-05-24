import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir_path = os.path.dirname(current_dir)

sys.path.insert(0, src_dir_path)
from evaluation import compute_metric_general

answer1 = r"calculate_IAA/answer1.txt"
with open(answer1, 'r', encoding='utf-8') as outfile:
    answers = outfile.readlines()
    predicts = [answer.strip() for answer in answers]

answer2 = r"calculate_IAA/answer2.txt"
with open(answer2, 'r', encoding='utf-8') as expect_outfile:
    expect_answers = expect_outfile.readlines()
    expects = [answer.strip() for answer in expect_answers]

# Compute metrics
results = compute_metric_general(predicts, expects)


print('-------------results: ------------------')
print(results)

