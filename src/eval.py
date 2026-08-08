from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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


from rouge_score import rouge_scorer


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

