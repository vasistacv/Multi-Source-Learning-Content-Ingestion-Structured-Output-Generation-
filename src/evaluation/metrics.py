from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from dataclasses import dataclass
from loguru import logger

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import nltk

nltk.download('punkt', quiet=True)


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    details: Optional[Dict[str, Any]] = None


class ComprehensiveEvaluator:
    """Enterprise-grade evaluation suite for generated content quality."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        self.smoothing = SmoothingFunction()
    
    def evaluate_summary(
        self,
        generated_summary: str,
        reference_summary: str
    ) -> Dict[str, EvaluationResult]:
        """Evaluate summary quality using ROUGE metrics."""
        
        scores = self.rouge_scorer.score(reference_summary, generated_summary)
        
        return {
            "rouge1": EvaluationResult(
                metric_name="ROUGE-1",
                score=scores['rouge1'].fmeasure,
                details={
                    "precision": scores['rouge1'].precision,
                    "recall": scores['rouge1'].recall
                }
            ),
            "rouge2": EvaluationResult(
                metric_name="ROUGE-2",
                score=scores['rouge2'].fmeasure,
                details={
                    "precision": scores['rouge2'].precision,
                    "recall": scores['rouge2'].recall
                }
            ),
            "rougeL": EvaluationResult(
                metric_name="ROUGE-L",
                score=scores['rougeL'].fmeasure,
                details={
                    "precision": scores['rougeL'].precision,
                    "recall": scores['rougeL'].recall
                }
            )
        }
    
    def evaluate_answer(
        self,
        generated_answer: str,
        reference_answer: str
    ) -> Dict[str, EvaluationResult]:
        """Evaluate QA answer quality using BLEU and semantic metrics."""
        
        # BLEU score
        reference_tokens = nltk.word_tokenize(reference_answer.lower())
        generated_tokens = nltk.word_tokenize(generated_answer.lower())
        
        bleu_score = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            smoothing_function=self.smoothing.method1
        )
        
        # Exact match (normalized)
        exact_match = (
            generated_answer.strip().lower() == 
            reference_answer.strip().lower()
        )
        
        # Token overlap (F1)
        ref_set = set(reference_tokens)
        gen_set = set(generated_tokens)
        
        if len(gen_set) == 0:
            precision = 0
        else:
            precision = len(ref_set & gen_set) / len(gen_set)
        
        if len(ref_set) == 0:
            recall = 0
        else:
            recall = len(ref_set & gen_set) / len(ref_set)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "bleu": EvaluationResult(
                metric_name="BLEU",
                score=bleu_score
            ),
            "exact_match": EvaluationResult(
                metric_name="Exact Match",
                score=1.0 if exact_match else 0.0
            ),
            "token_f1": EvaluationResult(
                metric_name="Token F1",
                score=f1,
                details={"precision": precision, "recall": recall}
            )
        }
    
    def evaluate_flashcards(
        self,
        flashcards: List[Dict[str, str]],
        source_text: str
    ) -> Dict[str, EvaluationResult]:
        """Evaluate flashcard quality and coverage."""
        
        if not flashcards:
            return {
                "coverage": EvaluationResult("Coverage", 0.0),
                "quality": EvaluationResult("Quality", 0.0)
            }
        
        source_tokens = set(nltk.word_tokenize(source_text.lower()))
        
        # Coverage: How much of source content is covered
        covered_tokens = set()
        for fc in flashcards:
            q_tokens = nltk.word_tokenize(fc.get("question", "").lower())
            a_tokens = nltk.word_tokenize(fc.get("answer", "").lower())
            covered_tokens.update(q_tokens)
            covered_tokens.update(a_tokens)
        
        # Remove stop words for meaningful coverage
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                      'by', 'from', 'as', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'between', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when', 'where',
                      'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                      'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now'}
        
        meaningful_source = source_tokens - stop_words
        meaningful_covered = covered_tokens - stop_words
        
        coverage = len(meaningful_covered & meaningful_source) / max(len(meaningful_source), 1)
        
        # Quality metrics
        quality_scores = []
        for fc in flashcards:
            q = fc.get("question", "")
            a = fc.get("answer", "")
            
            # Question quality: Has question words?
            has_question_word = any(w in q.lower() for w in 
                                    ['what', 'why', 'how', 'when', 'where', 'which', 'who', 'explain', 'describe'])
            
            # Answer quality: Not too short, not too long
            answer_length = len(a.split())
            good_length = 5 <= answer_length <= 100
            
            # Relevance: Answer tokens appear in source
            a_tokens = set(nltk.word_tokenize(a.lower())) - stop_words
            relevance = len(a_tokens & meaningful_source) / max(len(a_tokens), 1)
            
            score = (0.3 * int(has_question_word) + 
                     0.3 * int(good_length) + 
                     0.4 * relevance)
            quality_scores.append(score)
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        return {
            "coverage": EvaluationResult(
                metric_name="Content Coverage",
                score=coverage,
                details={"covered_concepts": len(meaningful_covered & meaningful_source)}
            ),
            "quality": EvaluationResult(
                metric_name="Average Quality",
                score=avg_quality,
                details={"per_card_scores": quality_scores}
            ),
            "count": EvaluationResult(
                metric_name="Flashcard Count",
                score=float(len(flashcards))
            )
        }
    
    def evaluate_knowledge_graph(
        self,
        graph_data: Dict[str, Any],
        source_concepts: List[str]
    ) -> Dict[str, EvaluationResult]:
        """Evaluate knowledge graph quality."""
        
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        if not nodes:
            return {"graph_quality": EvaluationResult("Graph Quality", 0.0)}
        
        # Node coverage
        node_labels = set(n.get("label", "").lower() for n in nodes)
        source_set = set(c.lower() for c in source_concepts)
        
        coverage = len(node_labels & source_set) / max(len(source_set), 1)
        
        # Graph density
        max_edges = len(nodes) * (len(nodes) - 1) / 2
        density = len(edges) / max(max_edges, 1)
        
        # Connectivity (average degree)
        if nodes:
            avg_degree = 2 * len(edges) / len(nodes)
        else:
            avg_degree = 0
        
        return {
            "node_coverage": EvaluationResult(
                metric_name="Node Coverage",
                score=coverage,
                details={"total_nodes": len(nodes), "matched": len(node_labels & source_set)}
            ),
            "density": EvaluationResult(
                metric_name="Graph Density",
                score=min(density, 1.0)
            ),
            "connectivity": EvaluationResult(
                metric_name="Avg Connectivity",
                score=avg_degree
            )
        }
    
    def run_full_evaluation(
        self,
        pipeline_output: Dict[str, Any],
        source_text: str
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation on pipeline output."""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Evaluate summary
        if "summary" in pipeline_output:
            # For now, use source text first 500 chars as reference
            # In production, you'd have human-written summaries
            summary_metrics = self.evaluate_summary(
                pipeline_output["summary"],
                source_text[:500]
            )
            results["metrics"]["summary"] = {
                k: {"score": v.score, "details": v.details}
                for k, v in summary_metrics.items()
            }
        
        # Evaluate flashcards
        if "flashcards" in pipeline_output:
            fc_metrics = self.evaluate_flashcards(
                pipeline_output["flashcards"],
                source_text
            )
            results["metrics"]["flashcards"] = {
                k: {"score": v.score, "details": v.details}
                for k, v in fc_metrics.items()
            }
        
        # Evaluate knowledge graph
        if "knowledge_graph" in pipeline_output:
            kg_data = pipeline_output["knowledge_graph"]
            concepts = pipeline_output.get("key_concepts", [])
            
            kg_metrics = self.evaluate_knowledge_graph(kg_data, concepts)
            results["metrics"]["knowledge_graph"] = {
                k: {"score": v.score, "details": v.details}
                for k, v in kg_metrics.items()
            }
        
        # Calculate overall score
        all_scores = []
        for category in results["metrics"].values():
            for metric in category.values():
                if isinstance(metric.get("score"), (int, float)):
                    all_scores.append(metric["score"])
        
        results["overall_score"] = np.mean(all_scores) if all_scores else 0
        
        return results
    
    def save_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: Path
    ):
        """Save detailed evaluation report."""
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Also create a human-readable summary
        summary_path = output_path.with_suffix(".txt")
        with open(summary_path, "w") as f:
            f.write("EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {evaluation_results.get('timestamp', 'N/A')}\n")
            f.write(f"Overall Score: {evaluation_results.get('overall_score', 0):.2%}\n\n")
            
            for category, metrics in evaluation_results.get("metrics", {}).items():
                f.write(f"\n{category.upper()}\n")
                f.write("-" * 30 + "\n")
                for metric_name, data in metrics.items():
                    score = data.get("score", 0)
                    f.write(f"  {metric_name}: {score:.4f}\n")
        
        logger.info(f"Summary report saved to {summary_path}")
