import re
from dataclasses import dataclass
from .utils import is_answer_correct


@dataclass
class EvalFuncs:

    @staticmethod
    def gsm8k(df):
        n_correct = 0
        correct_ids = []
        incorrect_ids = []
        for idx, pred_str, ref_str in zip(df["idx"], df["generated"], df["response"]):
            try:
                pred = re.search(r"####\s*(.+)", pred_str).group(1).strip()
            except AttributeError:
                pred = "missing"

            ref = re.search(r"####\s*(.+)", ref_str).group(1).strip()

            if pred == ref:
                correct_ids.append(idx)
                n_correct += 1
            else:
                incorrect_ids.append(idx)

        accuracy = n_correct / len(df)

        report = {
            "metrics": {
                "accuracy": accuracy,
            },
            "incorrect_ids": incorrect_ids,
            "correct_ids": correct_ids,
        }

        return report

    @staticmethod
    def mathqa(df):
        n_correct = 0
        correct_ids = []
        incorrect_ids = []
        for idx, pred_str, ref_str in zip(df["idx"], df["generated"], df["response"]):
            try:
                pred = re.search(r"So the answer is\s*(.+)", pred_str).group(1).strip()
            except AttributeError:
                last_line = pred_str.strip().splitlines()[-1].strip()
                preds = re.findall(r"\b\(?[a-zA-Z]\)", last_line)
                pred = preds[0] if preds else "missing"

            ref = re.search(r"So the answer is\s*(.+)", ref_str).group(1).strip()

            pred = re.sub(r"[^a-zA-Z]+", "", pred)
            ref = re.sub(r"[^a-zA-Z]+", "", ref)

            if pred == ref:
                correct_ids.append(idx)
                n_correct += 1
            else:
                incorrect_ids.append(idx)

        accuracy = n_correct / len(df)

        report = {
            "metrics": {
                "accuracy": accuracy,
            },
            "incorrect_ids": incorrect_ids,
            "correct_ids": correct_ids,
        }

        return report

    @staticmethod
    def math(df):
        n_correct = 0
        correct_ids = []
        incorrect_ids = []
        for idx, pred_str, ref_str in zip(df["idx"], df["generated"], df["response"]):
            if is_answer_correct(pred_str, ref_str):
                correct_ids.append(idx)
                n_correct += 1
            else:
                incorrect_ids.append(idx)

        accuracy = n_correct / len(df)

        report = {
            "metrics": {
                "accuracy": accuracy,
            },
            "incorrect_ids": incorrect_ids,
            "correct_ids": correct_ids,
        }

        return report

    @staticmethod
    def svamp(df):
        n_correct = 0
        correct_ids = []
        incorrect_ids = []
        for idx, pred_str, ref_str in zip(df["idx"], df["generated"], df["response"]):
            if pred_str.strip() == ref_str.strip():
                correct_ids.append(idx)
                n_correct += 1
            else:
                incorrect_ids.append(idx)

        accuracy = n_correct / len(df)

        report = {
            "metrics": {
                "accuracy": accuracy,
            },
            "incorrect_ids": incorrect_ids,
            "correct_ids": correct_ids,
        }

        return report
