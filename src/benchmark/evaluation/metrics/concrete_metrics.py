from typing import Any, List, Union, Dict
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np
from .base_metrics import BaseMetric
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge import Rouge
from transformers import pipeline
import bert_score
from collections import Counter
import math
from scipy.stats import pearsonr, spearmanr
from seqeval.metrics import classification_report as seqeval_classification_report

class AccuracyMetric(BaseMetric):
    """
    Class for computing accuracy.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute accuracy.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., top_k).
        :return: Accuracy score.
        """
        top_k = kwargs.get("top_k", None)
        if top_k:
            raise NotImplementedError("Top-k accuracy not implemented")
        return accuracy_score(labels, predictions)

class PrecisionMetric(BaseMetric):
    """
    Class for computing precision.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute precision.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: Precision score.
        """
        average = kwargs.get("average", "weighted")
        zero_division = kwargs.get("zero_division", 0)
        return precision_score(labels, predictions, average=average, zero_division=zero_division)

class F1ScoreMetric(BaseMetric):
    """
    Class for computing F1 score.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute F1 score.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: F1 score.
        """
        average = kwargs.get("average", "weighted")
        zero_division = kwargs.get("zero_division", 0)
        return f1_score(labels, predictions, average=average, zero_division=zero_division)

class RecallMetric(BaseMetric):
    """
    Class for computing recall.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute recall.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options (e.g., average).
        :return: Recall score.
        """
        average = kwargs.get("average", "weighted")
        zero_division = kwargs.get("zero_division", 0)
        return recall_score(labels, predictions, average=average, zero_division=zero_division)

class BLEUScoreMetric(BaseMetric):
    """
    Class for computing BLEU score with enhanced options.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute BLEU score.

        :param predictions: Model predictions (list of tokens or string).
        :param labels: Ground truth labels (list of tokens or string).
        :param kwargs: Additional options:
            - weights: tuple of weights for n-grams (default: (0.25, 0.25, 0.25, 0.25))
            - smoothing: bool or SmoothingFunction (default: True)
            - corpus_level: whether to compute corpus-level BLEU (default: False)
        :return: BLEU score.
        """
        if not isinstance(labels, list) or not isinstance(predictions, list):
            raise TypeError("BLEU requires list of reference sentences")
        
        weights = kwargs.get("weights", (0.25, 0.25, 0.25, 0.25))
        smoothing = kwargs.get("smoothing", True)
        corpus_level = kwargs.get("corpus_level", False)
        
        if isinstance(smoothing, bool) and smoothing:
            smoothing = SmoothingFunction().method1
        
        if corpus_level:
            # For corpus-level BLEU, references should be list of lists
            return corpus_bleu([[ref] for ref in labels], predictions, 
                              weights=weights, smoothing_function=smoothing)
        else:
            # For sentence-level BLEU
            return sentence_bleu([labels], predictions, 
                               weights=weights, smoothing_function=smoothing)

class ROUGEScoreMetric(BaseMetric):
    """
    Class for computing ROUGE score with enhanced options.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> Dict[str, float]:
        """
        Compute ROUGE score.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options:
            - metrics: list of ROUGE metrics to compute (default: ['rouge-l'])
            - stats: list of stats to return (default: ['f'])
        :return: Dictionary of ROUGE scores.
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have same length")
            
        metrics = kwargs.get("metrics", ['rouge-l'])
        stats = kwargs.get("stats", ['f'])
        
        rouge = Rouge()
        scores = rouge.get_scores(predictions, labels, avg=True)
        
        result = {}
        for metric in metrics:
            for stat in stats:
                result[f"{metric}-{stat}"] = scores[metric][stat]
        
        return result

class PerplexityMetric(BaseMetric):
    """
    Class for computing perplexity with enhanced stability.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute perplexity.

        :param predictions: Model predictions (log probabilities recommended).
        :param labels: Ground truth labels.
        :param kwargs: Additional options:
            - is_log_probs: whether predictions are log probabilities (default: False)
            - epsilon: small value to avoid log(0) (default: 1e-10)
        :return: Perplexity score.
        """
        is_log_probs = kwargs.get("is_log_probs", False)
        epsilon = kwargs.get("epsilon", 1e-10)
        
        predictions = np.array(predictions, dtype=np.float32)
        
        if is_log_probs:
            log_probs = predictions
        else:
            # Clip probabilities to avoid log(0)
            predictions = np.clip(predictions, epsilon, 1.0)
            log_probs = np.log(predictions)
        
        # Calculate cross-entropy
        cross_entropy = -np.mean(log_probs)
        perplexity = np.exp(cross_entropy)
        
        return perplexity

class ExactMatchMetric(BaseMetric):
    """
    Class for computing exact match with normalization options.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute exact match.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options:
            - normalize: whether to normalize strings before comparison (default: False)
            - ignore_case: whether to ignore case (default: False)
            - ignore_punct: whether to ignore punctuation (default: False)
        :return: Exact match score (1.0 for match, 0.0 otherwise).
        """
        normalize = kwargs.get("normalize", False)
        ignore_case = kwargs.get("ignore_case", False)
        ignore_punct = kwargs.get("ignore_punct", False)
        
        pred = str(predictions)
        label = str(labels)
        
        if normalize:
            if ignore_case:
                pred = pred.lower()
                label = label.lower()
            if ignore_punct:
                pred = ''.join(c for c in pred if c not in '.,;!?\'"')
                label = ''.join(c for c in label if c not in '.,;!?\'"')
        
        return float(pred == label)

class METEORScoreMetric(BaseMetric):
    """
    Class for computing METEOR score (requires NLTK with METEOR).
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute METEOR score.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options:
            - alpha: parameter for relative importance of precision vs recall (default: 0.9)
            - beta: parameter for fragmentation penalty (default: 3.0)
            - gamma: fragmentation penalty constant (default: 0.5)
        :return: METEOR score.
        """
        from nltk.translate.meteor_score import meteor_score
        
        alpha = kwargs.get("alpha", 0.9)
        beta = kwargs.get("beta", 3.0)
        gamma = kwargs.get("gamma", 0.5)
        
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(labels, str):
            labels = [labels]
            
        # METEOR expects tokenized strings
        if not all(isinstance(p, list) for p in predictions):
            predictions = [p.split() for p in predictions]
        if not all(isinstance(l, list) for l in labels):
            labels = [l.split() for l in labels]
            
        return meteor_score([labels], predictions, alpha=alpha, beta=beta, gamma=gamma)

class JaccardSimilarityMetric(BaseMetric):
    """
    Class for computing Jaccard similarity between texts.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute Jaccard similarity.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options:
            - ngram: n-gram size (default: 1)
            - normalize: whether to normalize text (default: False)
        :return: Jaccard similarity score.
        """
        ngram = kwargs.get("ngram", 1)
        normalize = kwargs.get("normalize", False)
        
        def get_ngrams(text, n):
            if normalize:
                text = text.lower()
            words = text.split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            
        pred_ngrams = set(get_ngrams(str(predictions), ngram))
        label_ngrams = set(get_ngrams(str(labels), ngram))
        
        intersection = len(pred_ngrams & label_ngrams)
        union = len(pred_ngrams | label_ngrams)
        
        return intersection / union if union > 0 else 0.0

class SemanticSimilarityMetric(BaseMetric):
    """
    Class for computing semantic similarity using sentence embeddings.
    """
    def compute(self, predictions: Any, labels: Any, **kwargs: Any) -> float:
        """
        Compute semantic similarity.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options:
            - model: embedding model to use (default: 'all-MiniLM-L6-v2')
            - metric: similarity metric ('cosine', 'euclidean', 'dot') (default: 'cosine')
        :return: Semantic similarity score.
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model_name = kwargs.get("model", 'all-MiniLM-L6-v2')
        metric = kwargs.get("metric", 'cosine')
        
        model = SentenceTransformer(model_name)
        
        pred_emb = model.encode(str(predictions))
        label_emb = model.encode(str(labels))
        
        if metric == 'cosine':
            return cosine_similarity([pred_emb], [label_emb])[0][0]
        elif metric == 'euclidean':
            return 1 / (1 + np.linalg.norm(pred_emb - label_emb))
        elif metric == 'dot':
            return np.dot(pred_emb, label_emb)
        else:
            raise ValueError(f"Unknown metric: {metric}")

class DiversityMetric(BaseMetric):
    """
    Class for computing text diversity metrics.
    """
    def compute(self, texts: List[str], **kwargs: Any) -> Dict[str, float]:
        """
        Compute diversity metrics.

        :param texts: List of generated texts.
        :param kwargs: Additional options:
            - ngram: n-gram size for diversity (default: 1)
            - metric: which metrics to compute (default: ['distinct', 'entropy'])
        :return: Dictionary of diversity metrics.
        """
        ngram = kwargs.get("ngram", 1)
        metrics = kwargs.get("metrics", ['distinct', 'entropy'])
        
        def get_ngrams(text, n):
            words = text.split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            
        all_ngrams = []
        for text in texts:
            all_ngrams.extend(get_ngrams(text, ngram))
            
        counter = Counter(all_ngrams)
        total = len(all_ngrams)
        unique = len(counter)
        
        results = {}
        if 'distinct' in metrics:
            results[f'distinct_{ngram}'] = unique / total if total > 0 else 0.0
            
        if 'entropy' in metrics:
            probs = [count/total for count in counter.values()]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            results[f'entropy_{ngram}'] = entropy
            
        return results

class PearsonCorrelationMetric(BaseMetric):
    """
    Class for computing Pearson correlation coefficient.
    """
    def compute(self, predictions: List[float], labels: List[float], **kwargs: Any) -> float:
        """
        Compute Pearson correlation.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options.
        :return: Pearson correlation coefficient.
        """
        return pearsonr(predictions, labels)[0]

class SpearmanCorrelationMetric(BaseMetric):
    """
    Class for computing Spearman correlation coefficient.
    """
    def compute(self, predictions: List[float], labels: List[float], **kwargs: Any) -> float:
        """
        Compute Spearman correlation.

        :param predictions: Model predictions.
        :param labels: Ground truth labels.
        :param kwargs: Additional options.
        :return: Spearman correlation coefficient.
        """
        return spearmanr(predictions, labels)[0]

class SequenceLabelingMetrics(BaseMetric):
    """
    Class for computing sequence labeling metrics (NER, POS tagging, etc.)
    """
    def compute(self, predictions: List[List[str]], labels: List[List[str]], **kwargs: Any) -> Dict[str, float]:
        """
        Compute sequence labeling metrics.

        :param predictions: Model predictions (list of lists of tags).
        :param labels: Ground truth labels (list of lists of tags).
        :param kwargs: Additional options:
            - scheme: tagging scheme (e.g., 'IOB2', 'BIOES') (default: None)
            - mode: 'strict' or 'default' (default: 'default')
        :return: Dictionary of metrics (precision, recall, f1).
        """
        scheme = kwargs.get("scheme", None)
        mode = kwargs.get("mode", "default")
        
        report = seqeval_classification_report(labels, predictions, mode=mode, scheme=scheme, output_dict=True)
        
        # Flatten the report to get overall metrics
        if 'micro avg' in report:
            return {
                'precision': report['micro avg']['precision'],
                'recall': report['micro avg']['recall'],
                'f1': report['micro avg']['f1-score']
            }
        else:
            return {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            }
        
class FactualConsistencyMetric(BaseMetric):
    """
    Computes factual consistency between answer and context.
    """
    def __init__(self):
        self.qa_pipeline = pipeline(
            "question-answering",
            model="valhalla/squad-small-finetuned-bert-base-uncased"
        )

    def compute(self, 
               answers: List[str], 
               contexts: List[str], 
               **kwargs: Any) -> float:
        """
        Compute factual consistency score.
        
        :param answers: List of generated answers
        :param contexts: List of source contexts
        :return: Average consistency score [0-1]
        """
        scores = []
        for ans, ctx in zip(answers, contexts):
            result = self.qa_pipeline(
                question=ans,
                context=ctx,
                max_answer_len=100
            )
            scores.append(1.0 if result['score'] > 0.5 else 0.0)
        return np.mean(scores)

class DistinctNGramMetric(BaseMetric):
    """
    Computes distinct n-gram metrics for text diversity.
    """
    def compute(self, 
               texts: List[str], 
               **kwargs: Any) -> Dict[str, float]:
        """
        Compute diversity metrics.
        
        :param texts: List of generated texts
        :param kwargs: ngram sizes (default [1,2])
        :return: Dictionary of distinct-n scores
        """
        ngrams = kwargs.get("ngrams", [1, 2])
        results = {}
        
        for n in ngrams:
            all_ngrams = []
            for text in texts:
                words = text.split()
                ngrams_list = [' '.join(words[i:i+n]) 
                              for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams_list)
            
            unique = len(set(all_ngrams))
            total = len(all_ngrams)
            results[f"distinct_{n}"] = unique / total if total > 0 else 0.0
            
        return results

class WordEntropyMetric(BaseMetric):
    """
    Computes word-level entropy for text generation.
    """
    def compute(self, 
               texts: List[str], 
               **kwargs: Any) -> float:
        """
        Compute vocabulary entropy.
        
        :param texts: List of generated texts
        :return: Entropy value
        """
        word_counts = Counter()
        total_words = 0
        
        for text in texts:
            words = text.split()
            word_counts.update(words)
            total_words += len(words)
            
        probs = [count/total_words for count in word_counts.values()]
        return -sum(p * math.log(p) for p in probs if p > 0)
    
class BERTScoreMetric(BaseMetric):
    """
    Computes BERTScore for text generation evaluation.
    """
    def compute(self, 
                predictions: List[str], 
                references: List[str], 
                **kwargs: Any) -> Dict[str, float]:
        """
        Compute BERTScore between predictions and references.
        
        :param predictions: List of generated texts
        :param references: List of reference texts
        :param kwargs: Additional options (lang, model_type)
        :return: Dictionary with precision, recall, and F1 scores
        """
        lang = kwargs.get("lang", "en")
        model_type = kwargs.get("model_type", "bert-base-uncased")
        
        P, R, F1 = bert_score.score(
            cands=predictions,
            refs=references,
            lang=lang,
            model_type=model_type,
            verbose=False
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }

class ToxicityScoreMetric(BaseMetric):
    """
    Computes toxicity score using pre-trained classifier.
    """
    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None
        )

    def compute(self, 
               texts: List[str], 
               **kwargs: Any) -> Dict[str, float]:
        """
        Compute toxicity scores for input texts.
        
        :param texts: List of texts to analyze
        :return: Dictionary with toxicity probabilities
        """
        results = self.classifier(texts)
        return {
            "toxicity": np.mean([res[0]['score'] for res in results]),
            "severe_toxicity": np.mean([res[1]['score'] for res in results])
        }