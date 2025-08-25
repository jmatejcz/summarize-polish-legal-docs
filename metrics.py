import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from bert_score import score


def calculate_metrics(reference_text, summary_text):
    """
    Calculate ROUGE, ROUGE-L, BLEU, METEOR, and BERTScore between a reference text and a summary text.

    Args:
        reference_text (str): The reference or "gold standard" text
        summary_text (str): The text to be evaluated

    Returns:
        dict: Dictionary containing the calculated metrics
    """
    results = {}

    reference_text = str(reference_text)
    summary_text = str(summary_text)

    if not reference_text or not summary_text:
        return {
            "rouge-l": 0.0,
            "bleu": 0.0,
            "meteor": 0.0,
            "bertscore": 0.0,
        }

    rouge = Rouge()
    rouge_scores = rouge.get_scores(summary_text, reference_text)[0]

    results["rouge-l"] = rouge_scores["rouge-l"]["f"]

    reference_tokens = nltk.word_tokenize(reference_text)
    summary_tokens = nltk.word_tokenize(summary_text)

    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [reference_tokens],
        summary_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),  # Equal weights for 1-4 grams
        smoothing_function=smoothing,
    )
    results["bleu"] = bleu_score

    meteor = meteor_score([reference_tokens], summary_tokens)
    results["meteor"] = meteor

    P, R, F1 = score(
        [summary_text],
        [reference_text],
        model_type="allegro/herbert-base-cased",  # Polish BERT model
        num_layers=9,  # Default layer for BERT-base
        rescale_with_baseline=True,
        lang="pl",
        verbose=False,
    )
    results["bertscore"] = F1.item()  # Extract F1 score as float

    return results
