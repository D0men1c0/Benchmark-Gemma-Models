import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import math
from pathlib import Path

import torch

# Modules to test
from src.benchmark.evaluation.metrics.concrete_metrics import (
    AccuracyMetric,
    BERTScoreMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    ExactMatchMetric,
    BLEUScoreMetric,
    ROUGEScoreMetric,
    PerplexityMetric,
    METEORScoreMetric,
    JaccardSimilarityMetric,
    SemanticSimilarityMetric,
    DistinctNGramMetric,
    WordEntropyMetric,
    PearsonCorrelationMetric,
    SpearmanCorrelationMetric,
    SequenceLabelingMetrics,
    FactualConsistencyMetric,
    ToxicityScoreMetric,
    NoveltyScoreMetric,
    CustomScriptMetric
)
from src.benchmark.evaluation.metrics.base_metrics import BaseMetric

# --- Test AccuracyMetric ---
def test_accuracy_metric():
    metric = AccuracyMetric()
    metric.update_state(["A", "B", "C"], ["A", "B", "D"])
    assert metric.result() == 2/3
    metric.reset_state()
    assert metric.result() == 0.0
    metric.update_state([], [])
    assert metric.result() == 0.0

# --- Tests for SklearnScoreMetric derived classes ---
@pytest.mark.parametrize("metric_class, score_func_name, expected_value", [
    (PrecisionMetric, "precision_score", 0.5), # (2 TP) / (2 TP + 2 FP assuming specific setup)
    (RecallMetric, "recall_score", 0.5),    # (2 TP) / (2 TP + 2 FN assuming specific setup)
    (F1ScoreMetric, "f1_score", 0.5),        # Based on P and R
])
def test_sklearn_metrics(metric_class, score_func_name, expected_value):
    metric = metric_class()
    # Example: binary classification
    # Preds: [1, 0, 1, 0], Labels: [1, 1, 0, 0]
    # TP=1 (at index 0), FP=1 (at index 2), TN=1 (at index 3), FN=1 (at index 1)
    # Precision for class 1: TP / (TP+FP) = 1 / (1+1) = 0.5
    # Recall for class 1: TP / (TP+FN) = 1 / (1+1) = 0.5
    # F1 for class 1: 2 * (0.5 * 0.5) / (0.5+0.5) = 0.5
    # These are for class 1. 'weighted' average would consider class 0 as well.
    # For simplicity, let's assume a micro/macro average or specific class.
    # To make it simple and predictable for 'weighted' average:
    predictions = [1, 0, 1, 0, 1, 1]
    labels    = [1, 1, 0, 0, 1, 0]
    # Class 1: TP=2 (idx 0, 4), FP=2 (idx 2, 5)
    # Class 0: TP=1 (idx 3), FP=1 (idx 1)
    # Weighted Precision: ( (2/(2+2))*3 + (1/(1+1))*3 ) / 6 = (0.5*3 + 0.5*3)/6 = (1.5+1.5)/6 = 3/6 = 0.5
    # Weighted Recall: ( (2/(2+1))*3 + (1/(1+2))*3 ) / 6 = ((2/3)*3 + (1/3)*3)/6 = (2+1)/6 = 3/6 = 0.5
    # Weighted F1: ( (2*(0.5*2/3)/(0.5+2/3))*3 + (2*(0.5*1/3)/(0.5+1/3))*3 ) / 6
    #   F1_class1 = 2 * (P1*R1) / (P1+R1) = 2 * (0.5 * 2/3) / (0.5 + 2/3) = 2 * (1/3) / (7/6) = 2/3 * 6/7 = 4/7
    #   F1_class0 = 2 * (P0*R0) / (P0+R0) = 2 * (0.5 * 1/3) / (0.5 + 1/3) = 2 * (1/6) / (5/6) = 1/3 * 6/5 = 2/5
    # Weighted F1 = ( (4/7)*3 + (2/5)*3 ) / 6 = (12/7 + 6/5)/6 = ((60+42)/35)/6 = (102/35)/6 = 17/35 approx 0.4857
    # Let's use simpler data for predictable weighted average of 0.5
    predictions_simple = [1, 0]
    labels_simple      = [1, 0]
    metric.set_options(average="weighted", zero_division=0) # Default for the metric
    metric.update_state(predictions_simple, labels_simple)
    # P_c1 = 1/1=1, R_c1=1/1=1, F1_c1=1. P_c0=1/1=1, R_c0=1/1=1, F1_c0=1. Weighted F1 = 1.0
    # This is not 0.5. Let's adjust data or expectation.
    # For P, R, F1 to be 0.5 with weighted average:
    # Class 1: 1 pred, 1 label, correct -> P=1, R=1, F1=1
    # Class 0: 1 pred, 1 label, incorrect -> P=0, R=0, F1=0
    # Weighted = (1*count1 + 0*count0) / (count1+count0) = 1*1 / 2 = 0.5
    metric.reset_state()
    metric.update_state([1, 1], [1, 0]) # Preds, Labels
    # P(1): 1/(1+1)=0.5, R(1): 1/1=1, F1(1) = 2*0.5*1 / 1.5 = 0.666
    # P(0): 0/0=0, R(0): 0/1=0, F1(0)=0
    # Weighted F1 = (0.666*1 + 0*1)/2 = 0.333. Still not 0.5.
    # The test setup for sklearn metrics needs careful thought for weighted avg.
    # Let's test micro average or simplify.
    # If we use average="binary" and pos_label=1
    metric.reset_state()
    metric.set_options(average="binary", pos_label=1, zero_division=0)
    metric.update_state([1, 0, 1, 0], [1, 1, 0, 0]) # P:1/2=0.5, R:1/2=0.5, F1:0.5
    assert metric.result() == pytest.approx(expected_value, 0.01)

# --- Test ExactMatchMetric ---
def test_exact_match_metric():
    metric = ExactMatchMetric()
    metric.update_state(["hello world"], ["hello world"])
    assert metric.result() == 1.0
    metric.update_state(["test"], ["Test"])
    assert metric.result() == 0.5 # 1 match / 2 total

    metric.reset_state()
    metric.set_options(normalize=True, ignore_case=True, ignore_punct=True)
    metric.update_state(["Test, Test!"], ["test test"])
    assert metric.result() == 1.0

# --- Test BLEUScoreMetric ---
@patch('nltk.download') # To prevent actual downloads during tests
def test_bleu_score_metric(mock_nltk_download):
    metric = BLEUScoreMetric()
    # nltk.download usually happens if resources not found. Mock it.
    with patch('nltk.data.find', return_value=True): # Assume resources are found
        metric.update_state(["hello there general kenobi"], ["hello there general kenobi"])
        metric.update_state(["this is another test"], ["this is a different test"])
        # Exact match for first, some overlap for second
        # corpus_bleu is sensitive, exact result depends on smoothing, n-grams etc.
        # Let's test for non-zero and specific cases.
        result1 = metric.result() # Default BLEU-4, method1 smoothing
        assert result1 > 0.0 and result1 <= 1.0

        metric.reset_state()
        metric.update_state(["a b c d"], ["a b c d"])
        assert metric.result() == pytest.approx(1.0, 0.001) # Perfect match should be close to 1

        metric.reset_state()
        metric.update_state(["completely different text"], ["nothing in common here"])
        result_no_overlap = metric.result()
        assert result_no_overlap < 0.1 # Expect very low score

        metric.reset_state()
        metric.set_options(smoothing="method1", weights=(1,0,0,0)) # BLEU-1, method1 smoothing
        metric.update_state(["the cat sat on the mat"], ["the cat sat on the mat in the room"]) # More overlap
        # Hyp: "the cat sat on the mat" (6) ; Ref: "the cat sat on the mat in the room" (9)
        # P1 = 6/6 = 1. BP = exp(1 - 9/6) = exp(1-1.5) = exp(-0.5) ~ 0.606
        # BLEU-1 ~ 0.606
        bleu1_score_new_example = metric.result()
        assert bleu1_score_new_example > 0.5 and bleu1_score_new_example < 0.7

# --- Test ROUGEScoreMetric ---
def test_rouge_score_metric():
    metric = ROUGEScoreMetric()
    metric.update_state(["hello world"], ["hello world"])
    results = metric.result()
    assert "rouge-l_f" in results
    assert results["rouge-l_f"] == pytest.approx(1.0)

    metric.reset_state()
    metric.set_options(metrics=['rouge1', 'rouge2'], stats=['p', 'r'])
    metric.update_state(["the quick brown fox"], ["the quick brown dog"])
    results2 = metric.result()
    assert "rouge1_p" in results2 and "rouge1_r" in results2
    assert "rouge2_p" in results2 and "rouge2_r" in results2
    assert results2["rouge1_p"] > 0.0 and results2["rouge1_r"] > 0.0

# --- Test PerplexityMetric ---
def test_perplexity_metric():
    metric = PerplexityMetric()
    # Perplexity is exp(-avg_log_likelihood)
    # If avg_log_likelihood is -2, perplexity is exp(2) approx 7.389
    metric.update_state([-1.0, -2.0, -3.0], [None, None, None]) # Labels are ignored
    avg_log_prob = (-1.0 -2.0 -3.0) / 3.0 # = -2.0
    expected_perplexity = math.exp(-avg_log_prob)
    assert metric.result() == pytest.approx(expected_perplexity)

# --- Test METEORScoreMetric ---
@patch('src.benchmark.evaluation.metrics.concrete_metrics.nltk_meteor_score')
def test_meteor_score_metric(mock_nltk_meteor_score_actual_call):
    metric = METEORScoreMetric()
    with patch.object(metric, '_ensure_nltk_resources', return_value=True):

        def meteor_side_effect(references, hypothesis, **kwargs):
            # Use isinstance to handle both str and list inputs
            hyp_str = " ".join(hypothesis) if isinstance(hypothesis, list) else str(hypothesis)

            if hyp_str == "this is a test":
                return 0.93
            elif hyp_str == "exact match":
                return 1.0
            return 0.0 # Fallback

        mock_nltk_meteor_score_actual_call.side_effect = meteor_side_effect

        metric.update_state(["this is a test"], ["this is the test"])
        result = metric.result()
        assert result == pytest.approx(0.93)

        metric.reset_state()
        metric.update_state(["exact match"], ["exact match"])
        assert metric.result() == pytest.approx(1.0)

# --- Test JaccardSimilarityMetric ---
def test_jaccard_similarity_metric():
    metric = JaccardSimilarityMetric()
    metric.set_options(ngram=1, normalize=True)
    metric.update_state(["apple banana orange"], ["apple orange grape"])
    # Intersection: {apple, orange} (2)
    # Union: {apple, banana, orange, grape} (4)
    # Jaccard: 2/4 = 0.5
    assert metric.result() == 0.5

# --- Test SemanticSimilarityMetric ---
@patch('sentence_transformers.SentenceTransformer')
def test_semantic_similarity_metric(mock_sentence_transformer_cls):
    mock_model_instance = MagicMock()
    # Simulate encode to return fixed embeddings for predictability
    # Embedding for "good"
    emb_good = np.array([0.1, 0.2, 0.3])
    # Embedding for "great" (similar to good)
    emb_great = np.array([0.12, 0.22, 0.32])
    # Embedding for "bad" (different from good)
    emb_bad = np.array([0.8, 0.1, 0.15])

    def encode_side_effect(sentences):
        if isinstance(sentences, str): # Single sentence
            if sentences == "good": return emb_good
            if sentences == "great": return emb_great
            if sentences == "bad": return emb_bad
        # Simplified: assumes single sentence per call for this test
        return np.array([0.0, 0.0, 0.0]) # Default fallback

    mock_model_instance.encode.side_effect = encode_side_effect
    mock_sentence_transformer_cls.return_value = mock_model_instance

    metric = SemanticSimilarityMetric()
    metric.set_options(model='mock-model', metric_type='cosine') # metric_type was 'metric'
    
    metric.update_state(["good"], ["great"]) # Should be high similarity
    score1 = metric.result()
    
    metric.reset_state()
    metric.update_state(["good"], ["bad"]) # Should be lower similarity
    score2 = metric.result()
    
    assert score1 > score2
    assert score1 > 0.9 # Expect high cosine similarity for similar mock embeddings
    mock_sentence_transformer_cls.assert_called_with('mock-model')


# --- Test DistinctNGramMetric ---
def test_distinct_ngram_metric():
    metric = DistinctNGramMetric()
    metric.set_options(ngrams=[1, 2])
    preds = ["a b c", "a b c d", "x y z"]
    metric.update_state(preds, [None]*len(preds))
    results = metric.result()
    # Distinct-1: a,b,c,d,x,y,z (7 unique) / Total words: 3+4+3=10. distinct_1 = 7/10 = 0.7
    # Distinct-2: "a b", "b c", "c d", "x y", "y z" (5 unique) / Total 2-grams: 2+3+2=7. distinct_2 = 5/7
    assert results["distinct_1"] == pytest.approx(7/10)
    assert results["distinct_2"] == pytest.approx(5/7)

# --- Test WordEntropyMetric ---
def test_word_entropy_metric():
    metric = WordEntropyMetric()
    preds = ["a a b b", "a c"]
    # Words: a (3), b (2), c (1). Total = 6
    # P(a)=3/6=0.5, P(b)=2/6=1/3, P(c)=1/6
    # Entropy = -(0.5*log2(0.5) + (1/3)*log2(1/3) + (1/6)*log2(1/6))
    #         = - (0.5*(-1) + (1/3)*(-1.585) + (1/6)*(-2.585))
    #         = - (-0.5 - 0.5283 - 0.4308) = 1.4591
    metric.update_state(preds, [None]*len(preds))
    assert metric.result() == pytest.approx(1.4591, 0.001)

# --- Test CorrelationMetrics ---
@pytest.mark.parametrize("metric_class, scipy_func_name", [
    (PearsonCorrelationMetric, "pearsonr"),
    (SpearmanCorrelationMetric, "spearmanr"),
])
def test_correlation_metrics(metric_class, scipy_func_name):
    metric = metric_class()
    preds = [1, 2, 3, 4, 5]
    labels = [1.1, 2.2, 2.8, 4.1, 5.3] # Strong positive correlation
    metric.update_state(preds, labels)
    result = metric.result()
    assert result > 0.9 and result <= 1.0

    metric.reset_state()
    preds_neg = [1, 2, 3, 4, 5]
    labels_neg = [5, 4, 3, 2, 1]
    metric.update_state(preds_neg, labels_neg)
    result_neg = metric.result()
    assert result_neg < -0.9 and result_neg >= -1.0


# --- Test SequenceLabelingMetrics ---
@patch('src.benchmark.evaluation.metrics.concrete_metrics.seqeval_classification_report')
def test_sequence_labeling_metrics(mock_seqeval_report):
    metric = SequenceLabelingMetrics()
    mock_report_dict = {
        "LOC": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 2},
        "micro avg": {"precision": 0.6, "recall": 0.7, "f1-score": 0.65, "support": 10}
    }
    mock_seqeval_report.return_value = mock_report_dict
    
    preds = [["O", "B-LOC", "I-LOC"], ["O"]]
    labels = [["O", "B-LOC", "O"], ["B-LOC"]]
    metric.update_state(preds, labels)
    metric.set_options(average_type="micro avg")
    results = metric.result()
    
    assert results["overall_precision"] == 0.6
    assert results["overall_recall"] == 0.7
    assert results["overall_f1"] == 0.65
    mock_seqeval_report.assert_called_once_with(
        y_true=labels, y_pred=preds, mode="default", scheme=None, output_dict=True, zero_division=0
    )

# --- Test BERTScoreMetric ---
@patch('bert_score.score')
def test_bert_score_metric(mock_bert_score_fn):
    metric = BERTScoreMetric()
    # Mock return value of bert_score.score: (P, R, F1) tensors
    mock_p = torch.tensor([0.8, 0.9])
    mock_r = torch.tensor([0.7, 0.85])
    mock_f1 = torch.tensor([0.75, 0.87])
    mock_bert_score_fn.return_value = (mock_p, mock_r, mock_f1)
    
    preds = ["hello there", "general kenobi"]
    labels = ["hello there world", "general kenobi obiwan"]
    metric.set_options(lang="en", verbose=False, bertscore_batch_size=32)
    metric.update_state(preds, labels)
    results = metric.result()
    
    assert results["bertscore_precision"] == pytest.approx(mock_p.mean().item())
    assert results["bertscore_recall"] == pytest.approx(mock_r.mean().item())
    assert results["bertscore_f1"] == pytest.approx(mock_f1.mean().item())
    mock_bert_score_fn.assert_called_once_with(
        cands=preds, refs=labels, lang="en", verbose=False, idf=False, batch_size=32, device=None
    )

# --- Test FactualConsistencyMetric ---

@patch('src.benchmark.evaluation.metrics.concrete_metrics.pipeline')
@patch('torch.cuda.is_available') # Add this patch
def test_factual_consistency_metric(mock_cuda_available, mock_hf_pipeline): # Add mock_cuda_available
    mock_cuda_available.return_value = False # Control the return value, e.g., force CPU path for this test
    expected_device_for_pipeline = -1 # Corresponds to CPU

    mock_qa_pipeline_instance = MagicMock()
    mock_qa_pipeline_instance.return_value = {"score": 0.95, "answer": "some answer"}
    mock_hf_pipeline.return_value = mock_qa_pipeline_instance

    metric = FactualConsistencyMetric()
    # Set the device option if you want to override the default logic based on torch.cuda.is_available
    # For this test, we'll rely on the patched torch.cuda.is_available
    metric.set_options(model="mock-qa-model")

    metric.update_state(["What is the capital of France?"], ["Paris is the capital of France."])

    mock_hf_pipeline.assert_called_with(
        "question-answering", 
        model="mock-qa-model", 
        device=expected_device_for_pipeline # Use the controlled expected device
    )
    mock_qa_pipeline_instance.assert_called_with(question="What is the capital of France?", context="Paris is the capital of France.")
    assert metric.result() == 0.95

# --- Test ToxicityScoreMetric ---
@patch('src.benchmark.evaluation.metrics.concrete_metrics.pipeline')
def test_toxicity_score_metric(mock_hf_pipeline):
    mock_toxic_pipeline_instance = MagicMock()
    # Simulate pipeline output
    mock_toxic_pipeline_instance.return_value = [{'label': 'TOXICITY', 'score': 0.98}]
    mock_hf_pipeline.return_value = mock_toxic_pipeline_instance

    metric = ToxicityScoreMetric()
    metric.set_options(model="mock-toxic-model", target_label="TOXICITY")
    
    metric.update_state(["you are a bad example"], [None]) # Labels ignored
    
    mock_hf_pipeline.assert_called_with("text-classification", model="mock-toxic-model", device=pytest.approx(0 if torch.cuda.is_available() else -1))
    mock_toxic_pipeline_instance.assert_called_with("you are a bad example")
    assert metric.result() == {"toxicity": 0.98}

    # Test case where target label is different
    metric.reset_state()
    mock_toxic_pipeline_instance.return_value = [{'label': 'SEVERE_TOXICITY', 'score': 0.7}]
    metric.set_options(model="mock-toxic-model", target_label="SEVERE_TOXICITY")
    metric.update_state(["another bad one"], [None])
    assert metric.result() == {"severe_toxicity": 0.7}

# --- Test NoveltyScoreMetric ---
def test_novelty_score_metric():
    metric = NoveltyScoreMetric()
    metric.set_options(novelty_ngram_size=2)
    
    preds = [
        {"cleaned_text": "the cat sat on the mat", "word_count": 6},
        {"cleaned_text": "a new sentence with new words", "word_count": 6}
    ]
    # Label 1: "the cat was on the mat" -> novel: "cat sat", "sat on"
    # Gen: "the cat", "cat sat", "sat on", "on the", "the mat" (5)
    # Input: "the cat", "cat was", "was on", "on the", "the mat" (5)
    # Novel: "cat sat". gen_ngrams - input_ngrams. gen_ngrams = {"the cat", "cat sat", ...}
    # novelty for pred1 = 1 unique 2-gram ("cat sat") / 5 total 2-grams in pred1 = 1/5 = 0.2
    
    # Label 2: None -> novelty for pred2 = 1.0 (all ngrams are novel compared to None)
    labels = ["the cat was on the mat", None]
    
    metric.update_state(preds, labels)
    results = metric.result()
    
    # Calculation for first sample:
    # pred1_ngrams = {"the cat", "cat sat", "sat on", "on the", "the mat"} (5)
    # label1_ngrams = {"the cat", "cat was", "was on", "on the", "the mat"} (5)
    # novel_ngrams1 = pred1_ngrams - label1_ngrams = {"cat sat"} (1)
    # novelty1 = 1 / 5 = 0.2
    #
    # Calculation for second sample:
    # pred2_ngrams = {"a new", "new sentence", "sentence with", "with new", "new words"} (5)
    # label2 is None, so input_ngrams is empty or treated as such.
    # novel_ngrams2 = pred2_ngrams - set() = pred2_ngrams (5)
    # novelty2 = 5 / 5 = 1.0
    #
    # avg_novelty = (0.2 + 1.0) / 2 = 1.2 / 2 = 0.7
    assert results["novelty_score_avg"] == pytest.approx(0.7)

# --- Test CustomScriptMetric ---
@pytest.fixture
def dummy_metric_script(tmp_path: Path) -> Path:
    script_content = """
def my_init(options):
    return {"sum": 0, "count": 0, "custom_key": options.get("result_key_name", "my_score")}

def my_update(current_state, predictions, labels, options):
    for p, l in zip(predictions, labels):
        current_state["sum"] += p + l
        current_state["count"] += 1
    return current_state

def my_result(final_state, options):
    if final_state["count"] == 0:
        return {final_state["custom_key"]: 0.0}
    return {final_state["custom_key"]: final_state["sum"] / final_state["count"]}
"""
    script_file = tmp_path / "custom_metric_functions.py"
    script_file.write_text(script_content)
    return script_file

def test_custom_script_metric(dummy_metric_script: Path):
    metric = CustomScriptMetric()
    options = {
        "metric_script_path": str(dummy_metric_script),
        "metric_init_function_name": "my_init",
        "metric_update_function_name": "my_update",
        "metric_result_function_name": "my_result",
        "metric_script_args": {"result_key_name": "custom_avg_sum"}
    }
    metric.set_options(**options)
    metric.reset_state()
    
    metric.update_state([1, 2, 3], [10, 20, 30])
    # sum = (1+10) + (2+20) + (3+30) = 11 + 22 + 33 = 66
    # count = 3
    # avg = 66 / 3 = 22
    results = metric.result()
    assert results == {"custom_avg_sum": 22.0}

def test_custom_script_metric_missing_options():
    metric = CustomScriptMetric()
    with pytest.raises(ValueError, match="CustomScriptMetric requires 'metric_script_path' and all three function names"):
        metric.set_options(metric_script_path="path/to/script.py")

def test_custom_script_metric_file_not_found():
    metric = CustomScriptMetric()
    options = {
        "metric_script_path": "non_existent_script.py",
        "metric_init_function_name": "init",
        "metric_update_function_name": "update",
        "metric_result_function_name": "result",
    }
    with pytest.raises(FileNotFoundError):
        metric.set_options(**options)

def test_custom_script_metric_function_not_found(tmp_path: Path):
    script_content = "def actual_init_name(): pass"
    p = tmp_path / "script.py"
    p.write_text(script_content)
    metric = CustomScriptMetric()
    options = {
        "metric_script_path": str(p),
        "metric_init_function_name": "wrong_init_name", # This function does not exist
        "metric_update_function_name": "update",
        "metric_result_function_name": "result",
    }
    with pytest.raises(AttributeError, match="Function 'wrong_init_name' not found"):
        metric.set_options(**options)



# --- Test AccuracyMetric ---
def test_accuracy_metric():
    metric = AccuracyMetric()
    metric.update_state(["A", "B", "C"], ["A", "B", "D"])
    assert metric.result() == 2/3
    metric.reset_state()
    assert metric.result() == 0.0
    metric.update_state([], [])
    assert metric.result() == 0.0
    metric.update_state([1, 2, 3], [1, 2, 3])
    assert metric.result() == 1.0
    metric.update_state([1, 2, 3], [1, 2, 4]) # Total 6, 5 correct
    assert metric.result() == 5/6

# --- Tests for SklearnScoreMetric derived classes ---
@pytest.mark.parametrize("metric_class, preds, labels, options, expected_score", [
    (PrecisionMetric, [1, 0, 1, 0], [1, 1, 0, 0], {"average": "binary", "pos_label": 1}, 0.5), # P for class 1 = 1/2 = 0.5
    (PrecisionMetric, [1, 1], [1, 0], {"average": "micro"}, 0.5), # (1 TP) / (1 TP + 1 FP)
    (PrecisionMetric, [0, 1], [0, 1], {"average": "macro"}, 1.0), # P_0=1, P_1=1 -> macro_P=1
    (PrecisionMetric, [0, 0], [1, 1], {"average": "weighted", "zero_division": 0}, 0.0), # All incorrect for class 0, no class 1 preds
    (RecallMetric, [1, 0, 1, 0], [1, 1, 0, 0], {"average": "binary", "pos_label": 1}, 0.5), # R for class 1 = 1/2 = 0.5
    (RecallMetric, [1, 0], [1, 1], {"average": "micro"}, 0.5), # (1 TP) / (1 TP + 1 FN)
    (RecallMetric, [0, 1], [0, 1], {"average": "macro"}, 1.0),
    (F1ScoreMetric, [1, 0, 1, 0], [1, 1, 0, 0], {"average": "binary", "pos_label": 1}, 0.5), # F1 for class 1 = 0.5
    (F1ScoreMetric, [1, 0, 1, 0], [1, 1, 0, 0], {"average": "weighted"}, 0.5), # (F1_1 * 2 + F1_0 * 2) / 4 = (0.5*2 + 0.5*2)/4 = 0.5
    (F1ScoreMetric, [], [], {}, 0.0), # Empty case
])
def test_sklearn_metrics_extended(metric_class, preds, labels, options, expected_score):
    metric = metric_class()
    metric.set_options(**options)
    metric.update_state(preds, labels)
    assert metric.result() == pytest.approx(expected_score, 0.01)

def test_sklearn_metrics_zero_division_option():
    metric = PrecisionMetric()
    metric.set_options(average="binary", pos_label=1, zero_division=1)
    metric.update_state([0, 0], [1, 1]) # No positive predictions, TP=0, FP=0 -> P = 0/0
    assert metric.result() == 1.0 # Due to zero_division=1

    metric.set_options(average="binary", pos_label=1, zero_division=0)
    metric.update_state([0, 0], [1, 1])
    assert metric.result() == 0.0 # Due to zero_division=0

# --- Test ExactMatchMetric ---
def test_exact_match_metric_options():
    metric = ExactMatchMetric()
    # Test ignore_case
    metric.set_options(normalize=True, ignore_case=True, ignore_punct=False)
    metric.update_state(["Test!"], ["test!"])
    assert metric.result() == 1.0
    metric.update_state(["Test2"], ["tesT2"]) # Total 2, 2 matches
    assert metric.result() == 1.0

    # Test ignore_punct
    metric.reset_state()
    metric.set_options(normalize=True, ignore_case=False, ignore_punct=True)
    metric.update_state(["Test, Test!"], ["Test Test"])
    assert metric.result() == 1.0
    metric.update_state(["NoPunct"], ["NoPunct"]) # Total 2, 2 matches
    assert metric.result() == 1.0

    # Test all normalization
    metric.reset_state()
    # Explicitly set options for this part of the test to ensure no normalization
    metric.set_options(normalize=False, ignore_case=False, ignore_punct=False)
    metric.update_state(["Hello"], ["hello"])
    assert metric.result() == 0.0 # "Hello" != "hello" (case-sensitive)

    metric.reset_state()
    metric.set_options() # Call with no args to reset to default empty options
    metric.update_state(["Test!"], ["Test"]) # Should not match if punct is not ignored by default
    assert metric.result() == 0.0

# --- Test BLEUScoreMetric ---
@patch('nltk.download')
def test_bleu_score_metric_various_options(mock_nltk_download):
    metric = BLEUScoreMetric()
    with patch('nltk.data.find', return_value=True):
        # Test different weights (BLEU-1 to BLEU-4)
        metric.set_options(weights=(1, 0, 0, 0)) # BLEU-1
        metric.update_state(["the cat is on the mat"], ["the cat sat on the mat"])
        bleu1 = metric.result()
        assert bleu1 > 0.6 and bleu1 < 0.9 # Expect high for unigram overlap

        metric.reset_state()
        metric.set_options(weights=(0.5, 0.5, 0, 0)) # BLEU-2 (equal weight to 1 & 2-grams)
        metric.update_state(["the cat is on the mat"], ["the cat sat on the mat"])
        bleu2 = metric.result()
        assert bleu2 < bleu1 # Generally BLEU-n score decreases as n increases for non-perfect matches

        # Test invalid prediction type (should be skipped)
        metric.reset_state()
        metric.update_state([123], ["this is a string"]) # Invalid prediction
        metric.update_state(["valid pred"], ["valid ref"])
        assert len(metric._collected_predictions_tokens) == 1 # Only one valid pred
        assert metric.result() > 0 # Should compute based on the valid one

        # Test different smoothing methods
        metric.reset_state()
        metric.set_options(smoothing="method4")
        metric.update_state(["a b c"], ["x y z"]) # No overlap
        assert metric.result() >= 0.0 # Smoothing should prevent zero with some methods

# --- Test ROUGEScoreMetric ---
def test_rouge_score_metric_edge_cases():
    metric = ROUGEScoreMetric()
    # Empty predictions/labels (already tested in a way by default empty state)
    assert metric.result() == {"rouge-l_f": 0.0}

    # Predictions are empty strings
    metric.update_state(["", ""], ["a b c", "d e f"])
    results_empty_preds = metric.result() # Based on current logic, these pairs are skipped
    assert results_empty_preds["rouge-l_f"] == 0.0

    # Labels are empty strings (should also result in 0 for typical ROUGE)
    metric.reset_state()
    metric.update_state(["a b c", "d e f"], ["", ""])
    results_empty_labels = metric.result()
    assert results_empty_labels["rouge-l_f"] == 0.0

    # Test specific metric and stat
    metric.reset_state()
    metric.set_options(metrics=['rouge2'], stats=['p'])
    metric.update_state(["the quick brown fox jumps"], ["the quick brown fox"])
    # Hyp: "the quick", "quick brown", "brown fox", "fox jumps"
    # Ref: "the quick", "quick brown", "brown fox"
    # Matching 2-grams: 3. Precision = 3/4 = 0.75
    results_rouge2p = metric.result()
    assert "rouge2_p" in results_rouge2p
    assert results_rouge2p["rouge2_p"] == pytest.approx(0.75)
    assert "rouge-l_f" not in results_rouge2p # Check only requested metrics are there

    metric.reset_state()
    metric.set_options(metrics=['rougeLsum'], stats=['f', 'r']) # Test rougeLsum
    metric.update_state(["test sentence one. test sentence two."], ["test sentence one. another sentence."])
    results_lsum = metric.result()
    assert "rougeLsum_f" in results_lsum and "rougeLsum_r" in results_lsum

# --- Test PerplexityMetric ---
def test_perplexity_metric_edge_cases():
    metric = PerplexityMetric()
    # No elements
    assert metric.result() == float('inf')

    # Single element
    metric.update_state([-2.0], [None])
    assert metric.result() == pytest.approx(math.exp(2.0))

    # Zero log prob (highly unlikely, but mathematically max perplexity)
    metric.reset_state()
    metric.update_state([0.0], [None]) # Log prob of 0 -> prob of 1
    assert metric.result() == pytest.approx(math.exp(0.0)) # PPL = 1

# --- Test METEORScoreMetric ---
@patch('src.benchmark.evaluation.metrics.concrete_metrics.nltk_meteor_score')
def test_meteor_score_metric_empty_inputs(mock_nltk_meteor_score_actual_call):
    metric = METEORScoreMetric()
    with patch.object(metric, '_ensure_nltk_resources', return_value=True):
        # No updates
        assert metric.result() == 0.0

        # Empty pred, empty ref
        metric.update_state([""], [""])
        mock_nltk_meteor_score_actual_call.return_value = 0.0 # What it would likely return or do
        assert metric.result() == 0.0 # Handled by the loop for score calc

        # Valid pred, empty ref
        metric.reset_state()
        metric.update_state(["not empty"], [""])
        # NLTK meteor_score might raise error or return 0. Test assumes it returns 0 or is handled.
        mock_nltk_meteor_score_actual_call.return_value = 0.0
        assert metric.result() == 0.0

# --- Test JaccardSimilarityMetric ---
def test_jaccard_similarity_metric_options():
    metric = JaccardSimilarityMetric()
    # Test different n-gram size
    metric.set_options(ngram=2, normalize=True)
    metric.update_state(["the cat sat"], ["the cat sat on mat"])
    # 2-grams pred: {"the cat", "cat sat"}
    # 2-grams label: {"the cat", "cat sat", "sat on", "on mat"}
    # Intersection: 2, Union: 4. Jaccard = 2/4 = 0.5
    assert metric.result() == 0.5

    # Test no overlap
    metric.reset_state()
    metric.set_options(ngram=1, normalize=True)
    metric.update_state(["apple"], ["orange"])
    assert metric.result() == 0.0

    # Test empty strings
    metric.reset_state()
    metric.update_state([""], [""])
    assert metric.result() == 0.0
    metric.update_state(["a"], [""])
    assert metric.result() == 0.0 # Intersection 0 / Union 1 (or 0 if pred also empty)


# --- Test SemanticSimilarityMetric ---
@patch('sentence_transformers.SentenceTransformer')
def test_semantic_similarity_metric_different_types(mock_sentence_transformer_cls):
    mock_model_instance = MagicMock()
    emb1 = np.array([0.1, 0.2])
    emb2 = np.array([0.8, 0.9])
    mock_model_instance.encode.side_effect = [emb1, emb2, emb1, emb2] # For two calls to update_state
    mock_sentence_transformer_cls.return_value = mock_model_instance

    metric = SemanticSimilarityMetric()
    
    # Test Euclidean
    metric.set_options(model='mock-model', metric_type='euclidean')
    metric.update_state(["text1"], ["text2"])
    dist = np.linalg.norm(emb1 - emb2)
    expected_euclidean_sim = 1 / (1 + dist)
    assert metric.result() == pytest.approx(expected_euclidean_sim)

    # Test Dot product
    metric.reset_state()
    metric.set_options(model='mock-model', metric_type='dot')
    metric.update_state(["text1"], ["text2"])
    expected_dot_sim = np.dot(emb1, emb2)
    assert metric.result() == pytest.approx(expected_dot_sim)

    # Test invalid metric_type
    metric.reset_state()
    metric.set_options(model='mock-model', metric_type='invalid_type')
    metric.update_state(["text1"], ["text2"]) # Should log a warning and return 0 for the item
    assert metric.result() == 0.0 

# --- Test DistinctNGramMetric ---
def test_distinct_ngram_metric_empty_and_short():
    metric = DistinctNGramMetric()
    metric.set_options(ngrams=[1, 2, 3])
    
    # Empty predictions
    metric.update_state([], [])
    assert metric.result() == {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0}

    # Predictions shorter than n
    metric.reset_state()
    metric.update_state(["a b", "c"], [None, None]) # For n=3, these are too short
    results = metric.result()
    assert results["distinct_1"] > 0.0 # a, b, c / 3 words
    assert results["distinct_2"] > 0.0 # "a b" / 1 2-gram
    assert results["distinct_3"] == 0.0 # No 3-grams

# --- Test WordEntropyMetric ---
def test_word_entropy_metric_single_word_or_empty():
    metric = WordEntropyMetric()
    metric.update_state(["test"], [None]) # Single unique word
    # P(test)=1. Entropy = -(1*log2(1)) = 0
    assert metric.result() == 0.0

    metric.reset_state()
    metric.update_state([""], [None]) # Empty string
    assert metric.result() == 0.0
    
    metric.reset_state()
    metric.update_state(["test test test"], [None]) # Single unique word, repeated
    assert metric.result() == 0.0


# --- Test CorrelationMetrics ---
def test_correlation_metrics_not_enough_data():
    pearson_metric = PearsonCorrelationMetric()
    spearman_metric = SpearmanCorrelationMetric()

    # Only one data point
    pearson_metric.update_state([1], [1])
    spearman_metric.update_state([1], [1])
    assert pearson_metric.result() == 0.0
    assert spearman_metric.result() == 0.0
    
    # Non-numeric data
    pearson_metric.reset_state()
    spearman_metric.reset_state()
    pearson_metric.update_state(["a", "b"], [1, 2])
    spearman_metric.update_state(["a", "b"], [1, 2])
    assert pearson_metric.result() == 0.0
    assert spearman_metric.result() == 0.0

# --- Test SequenceLabelingMetrics ---
@patch('src.benchmark.evaluation.metrics.concrete_metrics.seqeval_classification_report')
def test_sequence_labeling_metrics_options_and_errors(mock_seqeval_report):
    metric = SequenceLabelingMetrics()
    
    # Test different scheme and mode
    mock_seqeval_report.return_value = {"micro avg": {"f1-score": 0.75}}
    metric.set_options(scheme="IOB2", mode="strict", average_type="micro avg")
    metric.update_state([["O"]], [["O"]])
    metric.result()
    mock_seqeval_report.assert_called_with(
        y_true=[["O"]], y_pred=[["O"]], mode="strict", scheme="IOB2", output_dict=True, zero_division=0
    )

    # Test when report doesn't have the specified average_type
    metric.reset_state()
    mock_seqeval_report.return_value = {"custom_avg": {"f1-score": 0.9}} # Missing 'micro avg'
    metric.set_options(average_type="micro avg") # But we ask for micro avg
    metric.update_state([["O"]], [["O"]])
    results = metric.result()
    assert "error" in results
    error_message = results["error"]
    assert "Requested average type" in error_message
    assert "'micro avg'" in error_message # Expected average type
    assert "not found in seqeval report" in error_message
    assert "Available top-level keys: ['custom_avg']" in error_message


# --- Test BERTScoreMetric ---
@patch('bert_score.score')
def test_bert_score_metric_options(mock_bert_score_fn):
    metric = BERTScoreMetric()
    mock_p, mock_r, mock_f1 = torch.tensor([0.8]), torch.tensor([0.7]), torch.tensor([0.75])
    mock_bert_score_fn.return_value = (mock_p, mock_r, mock_f1)
    
    metric.set_options(lang="fr", model_type="bert-base-multilingual-cased", idf=True, bertscore_batch_size=16, device="cpu")
    metric.update_state(["bonjour"], ["salut"])
    metric.result()
    
    mock_bert_score_fn.assert_called_with(
        cands=["bonjour"], refs=["salut"], lang="fr", model_type="bert-base-multilingual-cased", 
        verbose=False, idf=True, batch_size=16, device="cpu"
    )

    # Test error during score calculation
    metric.reset_state()
    mock_bert_score_fn.side_effect = Exception("BERTScore lib error")
    metric.update_state(["a"],["b"])
    results = metric.result()
    assert "error" in results
    assert "BERTScore lib error" in results["error"]


# --- Test FactualConsistencyMetric & ToxicityScoreMetric (HFcommonPipelineMetric) ---
@patch('src.benchmark.evaluation.metrics.concrete_metrics.pipeline')
@patch('torch.cuda.is_available')
def test_hf_common_pipeline_metric_init_failure(mock_cuda_available, mock_hf_pipeline_ctor):
    mock_cuda_available.return_value = False
    mock_hf_pipeline_ctor.side_effect = OSError("Failed to load model") # Simulate pipeline init error

    metric = ToxicityScoreMetric() # Or FactualConsistencyMetric
    metric.set_options(model="non-existent-model-for-error")
    
    # update_state should not raise error but log a warning and skip
    metric.update_state(["some text"], [None]) 
    assert metric.items_count == 0 # No items processed
    
    # result should return default error value
    if isinstance(metric, ToxicityScoreMetric):
         assert metric.result() == {metric._target_label_for_score.lower(): 0.0}
    else: # FactualConsistency or other that returns float
         assert metric.result() == 0.0


@patch('src.benchmark.evaluation.metrics.concrete_metrics.pipeline')
@patch('torch.cuda.is_available')
def test_hf_common_pipeline_metric_process_item_failure(mock_cuda_available, mock_hf_pipeline_ctor):
    mock_cuda_available.return_value = False
    
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.side_effect = Exception("Pipeline processing error for an item")
    mock_hf_pipeline_ctor.return_value = mock_pipeline_instance

    metric = ToxicityScoreMetric()
    metric.set_options(model="dummy-model-will-be-mocked") # Model name for options
    
    metric.update_state(["text1", "text2"], [None, None]) # Should try to process 2 items
    
    assert mock_pipeline_instance.call_count == 2 # Pipeline instance called for each item
    assert metric.items_count == 0 # No scores successfully added
    assert metric.result() == {metric._target_label_for_score.lower(): 0.0} # Default on error/no items


# --- Test NoveltyScoreMetric ---
def test_novelty_score_metric_edge_cases_and_options():
    metric = NoveltyScoreMetric()
    
    # Empty predictions list
    metric.update_state([], [])
    assert metric.result()["novelty_score_avg"] == 0.0

    # Predictions with empty cleaned_text
    metric.reset_state()
    metric.update_state([{"cleaned_text": "", "word_count":0}], [None])
    assert metric.result()["novelty_score_avg"] == 0.0

    # Different ngram_size
    metric.reset_state()
    metric.set_options(novelty_ngram_size=1)
    preds = [{"cleaned_text": "apple banana", "word_count": 2}]
    labels = ["apple orange"]
    # 1-grams pred: {apple, banana}
    # 1-grams label: {apple, orange}
    # Novel: {banana}. Novelty = 1/2 = 0.5
    metric.update_state(preds, labels)
    assert metric.result()["novelty_score_avg"] == pytest.approx(0.5)

    # Prediction is not a dict
    metric.reset_state()
    metric.update_state(["just a string"], [None]) # Should log warning and skip
    assert metric.result()["novelty_score_avg"] == 0.0
    assert metric.num_samples == 0


# --- Test CustomScriptMetric ---
def test_custom_script_metric_error_in_user_script(tmp_path: Path, dummy_metric_script: Path):
    metric_error_script_content = """
def my_init(options): return {}
def my_update(current_state, predictions, labels, options): raise ValueError("Intentional update error")
def my_result(final_state, options): return {"score": 1.0}
"""
    error_script_file = tmp_path / "error_metric_script.py"
    error_script_file.write_text(metric_error_script_content)

    metric = CustomScriptMetric()
    options = {
        "metric_script_path": str(error_script_file),
        "metric_init_function_name": "my_init",
        "metric_update_function_name": "my_update", # This will fail
        "metric_result_function_name": "my_result",
    }
    metric.set_options(**options)
    metric.reset_state()
    
    # update_state should catch the error and log it, not propagate
    metric.update_state([1], [1]) 
    # The state might not be updated as expected due to error in update_fn

    # result should then reflect that computation might be incomplete or based on initial state
    # The current CustomScriptMetric.result() returns {"error": "custom_metric_not_computed"}
    # if update_fn was not loaded, but if it was loaded and failed, state might be bad.
    # Let's test if it returns the error dict if state is bad or result_fn itself fails.

    result_val = metric.result() # Should call my_result, but state might be just {}
    assert result_val == {"score": 1.0} # my_result ignores state and just returns this.
                                        # A better test would be if my_result used the state and it was not updated.

    # Test error in result_fn
    metric_error_result_content = """
def my_init(options): return {"val": 10}
def my_update(current_state, predictions, labels, options): current_state["val"] += 1; return current_state
def my_result(final_state, options): raise ValueError("Intentional result error")
"""
    error_result_script_file = tmp_path / "error_result_script.py"
    error_result_script_file.write_text(metric_error_result_content)
    metric_err_res = CustomScriptMetric()
    options_err_res = {
        "metric_script_path": str(error_result_script_file),
        "metric_init_function_name": "my_init",
        "metric_update_function_name": "my_update",
        "metric_result_function_name": "my_result", # This will fail
    }
    metric_err_res.set_options(**options_err_res)
    metric_err_res.reset_state()
    metric_err_res.update_state([1],[1])
    actual_error_result = metric_err_res.result()
    assert "error" in actual_error_result
    assert "custom_metric_result_fn_failed" in actual_error_result["error"]
    assert "Intentional result error" in actual_error_result["error"]


# --- More Tests for ROUGEScoreMetric ---
def test_rouge_score_metric_non_string_inputs():
    metric = ROUGEScoreMetric()
    metric.set_options(metrics=['rouge1'], stats=['f'])
    # Test with integer inputs (should be converted to string)
    metric.update_state([123, 456], [123, 789])
    # Pred1: "123", Label1: "123" -> ROUGE1-F = 1.0
    # Pred2: "456", Label2: "789" -> ROUGE1-F = 0.0
    # Average ROUGE1-F = (1.0 + 0.0) / 2 = 0.5
    results = metric.result()
    assert "rouge1_f" in results
    assert results["rouge1_f"] == pytest.approx(0.5)

    # Test with None inputs (should be handled gracefully, likely as empty strings by str())
    metric.reset_state()
    metric.update_state([None, "test"], [None, "test"])
    results_none = metric.result()
    # Pair 1 (None, None) -> should be skipped or result in 0 by rouge lib for empty strings
    # Pair 2 ("test", "test") -> rouge1_f = 1.0
    # If first pair results in 0 or is skipped by valid_hyps/refs logic:
    assert results_none["rouge1_f"] == pytest.approx(1.0) # if only valid pair is considered
                                                       # or 0.5 if (0+1)/2 if empty treated as 0


# --- More Tests for DistinctNGramMetric ---
def test_distinct_ngram_metric_varied_ngrams():
    metric = DistinctNGramMetric()
    metric.set_options(ngrams=[1, 2, 3, 4]) # Test up to 4-grams
    preds = ["a b c d e", "a b c d e"] # Identical predictions
    metric.update_state(preds, [None]*len(preds))
    results = metric.result()
    # For identical predictions, distinct-n will be count_unique_n_grams / total_n_grams
    # "a b c d e": 1-grams: a,b,c,d,e (5 unique / 5 total in one string -> 1.0)
    # Corpus has two "a b c d e".
    # Total 1-grams: 10. Unique: a,b,c,d,e (5). distinct_1 = 5/10 = 0.5
    # Total 2-grams: 4+4=8. Unique: "a b", "b c", "c d", "d e" (4). distinct_2 = 4/8 = 0.5
    # Total 3-grams: 3+3=6. Unique: "a b c", "b c d", "c d e" (3). distinct_3 = 3/6 = 0.5
    # Total 4-grams: 2+2=4. Unique: "a b c d", "b c d e" (2). distinct_4 = 2/4 = 0.5
    assert results["distinct_1"] == pytest.approx(0.5)
    assert results["distinct_2"] == pytest.approx(0.5)
    assert results["distinct_3"] == pytest.approx(0.5)
    assert results["distinct_4"] == pytest.approx(0.5)

    metric.reset_state()
    metric.set_options(ngrams=[1])
    preds_all_same_word = ["test test test", "test test"]
    metric.update_state(preds_all_same_word, [None]*len(preds_all_same_word))
    # Total 1-grams: 5 ("test" x5). Unique 1-grams: 1 ("test"). distinct_1 = 1/5 = 0.2
    assert metric.result()["distinct_1"] == pytest.approx(0.2)