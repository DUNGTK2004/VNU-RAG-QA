import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, default_data_collator
import re

class QuestionAnsweringModel:
    def __init__(self, model_name="xlm-roberta-base"):
        """
        Initialize a Question Answering model for Vietnamese.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """

        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)

    def load_dataset(self, dataset_name="harouzie/vi_question_generation", split="train"):
        """
        Load the Vietnamese question generation dataset.

        Args:
            dataset_name (str): Name of the dataset on HuggingFace Hub
            split (str): Dataset split to load (train, validation, test)

        Returns:
            dataset: The loaded dataset
        """
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)
        return dataset

    def preprocess_data(self, dataset, max_length=512, stride=128):
        """
        Preprocess the dataset for question answering.

        Args:
            dataset: The dataset to preprocess
            max_length (int): Maximum sequence length
            stride (int): Stride for handling long contexts

        Returns:
            processed_dataset: The processed dataset
        """

        def parse_answer_str(answer_str: str) -> dict:
            if not answer_str or not isinstance(answer_str, str):
                return {
                    "text": [],
                    "answer_start": []
                }
            texts = re.findall(r"array\(\s*\['(.+?)'\]", answer_str)
            starts = re.findall(r"array\(\s*\[(\d+)\]\)", answer_str)

            if len(texts) == 0 or len(starts) == 0 or len(texts) != len(starts):
                return {
                    "text": [],
                    "answer_start": []
                }

            return {
                "text": texts,
                "answer_start": [int(x) for x in starts]
            }

        def preprocess_function(examples):
            questions = examples["question"]
            contexts = examples["context"]
            answers = examples["answers"]

            inputs = self.tokenizer(
                questions,
                contexts,
                max_length=max_length,
                truncation="only_second",
                stride=stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            # Map offset mappings to find start and end positions
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = answers[sample_idx]
                if isinstance(answer, str):
                    answer = parse_answer_str(answer)

                if not answer or not answer["answer_start"] or not answer["text"]:
                    start_positions.append(0)
                    end_positions.append(0)
                    continue

                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            return inputs

        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return processed_dataset

    def train(self, train_dataset, eval_dataset=None, output_dir="./qa_model", epochs=3, batch_size=8):
        """
        Train the question answering model.

        Args:
            train_dataset: Preprocessed training dataset
            eval_dataset: Preprocessed evaluation dataset
            output_dir (str): Directory to save the model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training

        Returns:
            trainer: The trained model trainer
        """

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch" if eval_dataset else "no",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            push_to_hub=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        return trainer

    def predict(self, question, context):
        """
        Answer a question given a context.

        Args:
            question (str): The question to answer
            context (str): The context containing the answer

        Returns:
            dict: The predicted answer with text, start position, and end position
        """
        import torch

        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        # Get the most likely beginning and end of answer
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)

        # Convert token indices to text
        answer_tokens = input_ids[0][start_idx:end_idx+1]
        answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

        return {
            "text": answer_text,
            "start": start_idx.item(),
            "end": end_idx.item()
        }

    def evaluate(self, eval_dataset):
        """
        Evaluate the model on a dataset.

        Args:
            eval_dataset: Dataset to evaluate on

        Returns:
            dict: Evaluation metrics
        """
        from datasets import load_metric
        import numpy as np

        metric = load_metric("squad")

        def compute_metrics(p):
            predictions, labels = p
            start_logits, end_logits = predictions

            # Get the most likely beginning and end of answer
            start_preds = np.argmax(start_logits, axis=1)
            end_preds = np.argmax(end_logits, axis=1)

            # Prepare predictions in the format expected by the metric
            predictions = [
                {"id": str(i), "prediction_text": self.tokenizer.decode(eval_dataset[i]["input_ids"][s:e+1], skip_special_tokens=True)}
                for i, (s, e) in enumerate(zip(start_preds, end_preds))
            ]

            # Prepare references in the format expected by the metric
            references = [
                {"id": str(i), "answers": {"text": [self.tokenizer.decode(eval_dataset[i]["input_ids"][s:e+1], skip_special_tokens=True)], "answer_start": [s]}}
                for i, (s, e) in enumerate(zip(eval_dataset["start_positions"], eval_dataset["end_positions"]))
            ]

            return metric.compute(predictions=predictions, references=references)

        return compute_metrics
