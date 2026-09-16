"""
Redaction evaluation - measures precision, recall, and F1 against ground truth.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from .detector import PIIDetector


class RedactionEvaluator:
    """
    Evaluates redaction quality by comparing detected spans against ground truth.
    """

    def __init__(self, detector: PIIDetector, overlap_threshold: float = 0.5):
        """
        Initialize the evaluator.

        Args:
            detector: The PIIDetector to evaluate.
            overlap_threshold: IoU threshold for considering a detection correct.
                0.5 means at least 50% overlap with ground truth.
        """
        self.detector = detector
        self.overlap_threshold = overlap_threshold

    def evaluate_dataset(
        self, dataset_path: Path
    ) -> Dict[str, Any]:
        """
        Evaluate detection on the full synthetic dataset.

        Args:
            dataset_path: Path to synthetic_notes.jsonl

        Returns:
            Evaluation results with overall and per-entity-type metrics.
        """
        # Load all documents
        documents = []
        with dataset_path.open("r") as f:
            for line in f:
                documents.append(json.loads(line))

        # Track predictions and ground truth per entity type
        stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for doc in documents:
            text = doc["text"]
            ground_truth = doc["entities"]

            # Detect entities
            predicted = self.detector.detect(text)

            # Match predictions to ground truth
            matches = self._match_spans(predicted, ground_truth)

            # Update statistics
            for entity_type in set(
                [e["type"] for e in ground_truth] + [e["type"] for e in predicted]
            ):
                gt_of_type = [e for e in ground_truth if e["type"] == entity_type]
                pred_of_type = [e for e in predicted if e["type"] == entity_type]

                matched_gt = set()
                matched_pred = set()

                for pred_idx, gt_idx in matches:
                    if (
                        predicted[pred_idx]["type"] == entity_type
                        and ground_truth[gt_idx]["type"] == entity_type
                    ):
                        matched_gt.add(gt_idx)
                        matched_pred.add(pred_idx)

                # True positives: matched predictions
                tp = len(matched_pred)
                # False positives: predictions with no match
                fp = len(pred_of_type) - tp
                # False negatives: ground truth with no match
                fn = len(gt_of_type) - len(matched_gt)

                stats[entity_type]["tp"] += tp
                stats[entity_type]["fp"] += fp
                stats[entity_type]["fn"] += fn

        # Compute metrics
        results = {"per_type": {}, "overall": {}}

        overall_tp = 0
        overall_fp = 0
        overall_fn = 0

        for entity_type, counts in stats.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            precision, recall, f1 = self._compute_metrics(tp, fp, fn)

            results["per_type"][entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

        # Overall metrics
        precision, recall, f1 = self._compute_metrics(overall_tp, overall_fp, overall_fn)
        results["overall"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": overall_tp,
            "fp": overall_fp,
            "fn": overall_fn,
        }

        results["config"] = {
            "confidence_threshold": self.detector.confidence_threshold,
            "overlap_threshold": self.overlap_threshold,
            "num_documents": len(documents),
        }

        return results

    def _match_spans(
        self, predicted: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        """
        Match predicted spans to ground truth spans using IoU threshold.

        Args:
            predicted: List of predicted entities.
            ground_truth: List of ground truth entities.

        Returns:
            List of (predicted_idx, ground_truth_idx) pairs for matches.
        """
        matches = []

        # Build a greedy matching: for each prediction, find best ground truth match
        used_gt = set()

        for pred_idx, pred in enumerate(predicted):
            best_iou = 0
            best_gt_idx = None

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt:
                    continue

                # Type must match
                if pred["type"] != gt["type"]:
                    continue

                # Compute IoU
                iou = self._compute_iou(
                    (pred["start"], pred["end"]), (gt["start"], gt["end"])
                )

                if iou > best_iou and iou >= self.overlap_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is not None:
                matches.append((pred_idx, best_gt_idx))
                used_gt.add(best_gt_idx)

        return matches

    @staticmethod
    def _compute_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
        """
        Compute Intersection over Union for two character spans.

        Args:
            span1: (start, end) tuple
            span2: (start, end) tuple

        Returns:
            IoU score [0, 1]
        """
        start1, end1 = span1
        start2, end2 = span2

        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)

        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union = union_end - union_start

        return intersection / union if union > 0 else 0

    @staticmethod
    def _compute_metrics(
        tp: int, fp: int, fn: int
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.

        Args:
            tp: True positives
            fp: False positives
            fn: False negatives

        Returns:
            (precision, recall, f1) tuple
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return precision, recall, f1


def evaluate_redaction(
    dataset_path: Path,
    confidence_threshold: float = 0.5,
    overlap_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate redaction on a dataset.

    Args:
        dataset_path: Path to synthetic_notes.jsonl
        confidence_threshold: Minimum confidence for detections
        overlap_threshold: IoU threshold for matching

    Returns:
        Evaluation results dictionary
    """
    detector = PIIDetector(confidence_threshold=confidence_threshold)
    evaluator = RedactionEvaluator(
        detector=detector, overlap_threshold=overlap_threshold
    )
    return evaluator.evaluate_dataset(dataset_path)
