from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import logging

# Setup basic logging
logging.basicConfig(level=logging.WARNING)

def compute_cosine_similarity(text1: str, text2: str) -> float:
    try:
        if not text1.strip() or not text2.strip():
            return 0.0

        vectorizer = TfidfVectorizer().fit([text1, text2])
        tfidf = vectorizer.transform([text1, text2])
        score = cosine_similarity(tfidf[0], tfidf[1])[0][0]
        return round(score, 4)
    except Exception as e:
        logging.warning(f"[Cosine Similarity Error] {e}")
        return 0.0

def compute_rouge_l(summary: str, reference: str) -> float:
    try:
        if not summary.strip() or not reference.strip():
            return 0.0

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        score = scorer.score(reference.strip(), summary.strip())['rougeL'].fmeasure
        return round(score, 4)
    except Exception as e:
        logging.warning(f"[ROUGE-L Error] {e}")
        return 0.0
