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

