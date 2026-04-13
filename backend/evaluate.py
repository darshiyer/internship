from bert_score import score as bert_score_fn
from sacrebleu.metrics import BLEU, CHRF

bleu = BLEU(effective_order=True)
chrf = CHRF()


def compute_bleu(hypothesis: str, reference: str) -> float:
    result = bleu.sentence_score(hypothesis, [reference])
    return round(result.score, 4)


def compute_chrf(hypothesis: str, reference: str) -> float:
    result = chrf.sentence_score(hypothesis, [reference])
    return round(result.score, 4)


def compute_bertscore(hypotheses: list[str], references: list[str], lang: str = "kn") -> dict:
    p_score, r_score, f1_score = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    return {
        "precision": round(p_score.mean().item(), 4),
        "recall": round(r_score.mean().item(), 4),
        "f1": round(f1_score.mean().item(), 4),
    }
