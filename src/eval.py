from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calculate_bleu(reference, prediction):
    reference_words = reference.split()
    prediction_words = prediction.split()

    if len(prediction_words) == 0:
        return 0.0

    smoothing = SmoothingFunction().method1

    score = sentence_bleu(
        [reference_words],
        prediction_words,
        smoothing_function=smoothing
    )

    return score


def calculate_rouge(reference, prediction):
    if len(prediction.strip()) == 0:
        return {
            "rouge1": 0.0,
            "rougeL": 0.0
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference, prediction)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }


# Corpus-level BLEU
def corpus_bleu_score(references, predictions):
    if len(references) == 0:
        raise ValueError("no references given")
    if len(references) != len(predictions):
        raise ValueError("references and predictions must be same length")

    list_of_references = [[ref.split()] for ref in references]
    hypotheses = [pred.split() for pred in predictions]

    smoothing = SmoothingFunction().method1
    return corpus_bleu(list_of_references, hypotheses, smoothing_function=smoothing)


# Corpus-level ROUGE is averaged across all examples
def corpus_rouge_score(references, predictions):
    if len(references) == 0:
        raise ValueError("no references given")
    if len(references) != len(predictions):
        raise ValueError("references and predictions must be same length")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rougeL_scores = []
    for ref, pred in zip(references, predictions):
        if len(pred.strip()) == 0:
            rouge1_scores.append(0.0)
            rougeL_scores.append(0.0)
            continue
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
    }


# main function compare.py calls to score one system
def evaluate_system(references, predictions, name="model"):
    num_empty = sum(1 for p in predictions if len(p.strip()) == 0)

    bleu = corpus_bleu_score(references, predictions)
    rouge = corpus_rouge_score(references, predictions)

    return {
        "name": name,
        "num_examples": len(references),
        "num_empty_predictions": num_empty,
        "bleu": bleu,
        "rouge1": rouge["rouge1"],
        "rougeL": rouge["rougeL"],
    }


def evaluate_all_systems(references, system_predictions):
    return [
        evaluate_system(references, preds, name=name)
        for name, preds in system_predictions.items()
    ]

import json
import csv
import re
from pathlib import Path


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    project_root = Path(__file__).resolve().parent.parent

    # Load LSTM results
    lstm_file = project_root / "lstm_results.json"
    lstm_data = load_json(lstm_file)
    lstm_predictions = lstm_data["predictions"]

    # The LSTM file already has the correct references
    references = [
        normalize_text(item["reference"])
        for item in lstm_predictions
    ]

    sources = [
        item["source"]
        for item in lstm_predictions
    ]

    system_predictions = {
        "LSTM": [
            normalize_text(item["prediction"])
            for item in lstm_predictions
        ]
    }

    # Load all LLM result files
    llm_files = {
        "LLM V1 0-shot": "llm_results_v1_0shot.json",
        "LLM V1 3-shot": "llm_results_v1_3shot.json",
        "LLM V2 0-shot": "llm_results_v2_0shot.json",
        "LLM V2 3-shot": "llm_results_v2_3shot.json",
    }

    llm_data = {}

    for name, filename in llm_files.items():
        path = project_root / "results" / filename
        data = load_json(path)
        predictions = data["predictions"]

        if len(predictions) != len(references):
            raise ValueError(
                f"{name} has a different number of predictions"
            )

        # Make sure the same test examples are being compared
        for i, item in enumerate(predictions):
            if item["source"] != sources[i]:
                raise ValueError(
                    f"Test examples do not match for {name} at index {i}"
                )

        system_predictions[name] = [
            normalize_text(item["prediction"])
            for item in predictions
        ]

        llm_data[name] = predictions

    # Calculate BLEU and ROUGE for every system
    results = evaluate_all_systems(
        references,
        system_predictions
    )

    print("\nEvaluation Results")
    print("-" * 75)

    print(
        f"{'System':<18}"
        f"{'BLEU':>12}"
        f"{'ROUGE-1':>12}"
        f"{'ROUGE-L':>12}"
        f"{'Empty':>10}"
    )

    for result in results:
        print(
            f"{result['name']:<18}"
            f"{result['bleu'] * 100:>12.2f}"
            f"{result['rouge1'] * 100:>12.2f}"
            f"{result['rougeL'] * 100:>12.2f}"
            f"{result['num_empty_predictions']:>10}"
        )

    # Save the summary table
    summary_file = project_root / "evaluation_summary.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "System",
            "BLEU",
            "ROUGE-1",
            "ROUGE-L",
            "Empty Predictions"
        ])

        for result in results:
            writer.writerow([
                result["name"],
                result["bleu"] * 100,
                result["rouge1"] * 100,
                result["rougeL"] * 100,
                result["num_empty_predictions"]
            ])

    # Save all systems side by side
    comparison_file = (
        project_root / "side_by_side_predictions.csv"
    )

    with open(
        comparison_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Source",
            "Reference",
            "LSTM",
            "LLM V1 0-shot",
            "LLM V1 3-shot",
            "LLM V2 0-shot",
            "LLM V2 3-shot"
        ])

        for i in range(len(references)):
            writer.writerow([
                sources[i],
                lstm_predictions[i]["reference"],
                lstm_predictions[i]["prediction"],
                llm_data["LLM V1 0-shot"][i]["prediction"],
                llm_data["LLM V1 3-shot"][i]["prediction"],
                llm_data["LLM V2 0-shot"][i]["prediction"],
                llm_data["LLM V2 3-shot"][i]["prediction"]
            ])

    print("\nSaved:")
    print(summary_file)
    print(comparison_file)


if __name__ == "__main__":
    main()

