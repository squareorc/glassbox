"""
Redaction pipeline - ties detection, redaction, and evaluation together.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .detector import PIIDetector
from .redactor import PIIRedactor
from .evaluate import evaluate_redaction


def run_redaction_pipeline(
    input_path: Path,
    output_path: Path,
    audit_log_path: Optional[Path] = None,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run the full redaction pipeline: detect → redact → save.

    Args:
        input_path: Path to raw synthetic_notes.jsonl
        output_path: Path to write redacted documents (JSONL)
        audit_log_path: Optional path to write audit log (JSONL)
        confidence_threshold: Minimum confidence for detections

    Returns:
        Summary statistics: number of documents processed, entities redacted, etc.
    """
    detector = PIIDetector(confidence_threshold=confidence_threshold)
    redactor = PIIRedactor()

    # Process documents
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_log_path:
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_entities_detected = 0
    total_entities_redacted = 0

    with input_path.open("r") as f_in, output_path.open("w") as f_out:
        audit_records = []

        for line in f_in:
            doc = json.loads(line)
            text = doc["text"]
            doc_id = doc["doc_id"]

            # Detect PII
            entities = detector.detect(text)
            total_entities_detected += len(entities)

            # Redact PII
            redacted_text, audit_log = redactor.redact(text, entities)
            total_entities_redacted += len(audit_log)

            # Write redacted document
            redacted_doc = {
                "doc_id": doc_id,
                "text": redacted_text,
                "original_text": text,  # Keep for debugging/evaluation
                "diagnosis": doc["diagnosis"],
                "medications": doc["medications"],
                "num_entities_redacted": len(audit_log),
            }
            f_out.write(json.dumps(redacted_doc) + "\n")

            # Collect audit log
            if audit_log_path:
                for entry in audit_log:
                    audit_records.append({"doc_id": doc_id, **entry})

            total_docs += 1

        # Write audit log
        if audit_log_path:
            with audit_log_path.open("w") as f_audit:
                for record in audit_records:
                    f_audit.write(json.dumps(record) + "\n")

    summary = {
        "num_documents": total_docs,
        "num_entities_detected": total_entities_detected,
        "num_entities_redacted": total_entities_redacted,
        "confidence_threshold": confidence_threshold,
        "output_path": str(output_path),
        "audit_log_path": str(audit_log_path) if audit_log_path else None,
    }

    return summary


def run_evaluation(
    dataset_path: Path,
    confidence_threshold: float = 0.5,
    overlap_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run evaluation on the redaction module.

    Args:
        dataset_path: Path to synthetic_notes.jsonl with ground truth
        confidence_threshold: Minimum confidence for detections
        overlap_threshold: IoU threshold for matching

    Returns:
        Evaluation results with precision, recall, F1 per entity type
    """
    return evaluate_redaction(
        dataset_path=dataset_path,
        confidence_threshold=confidence_threshold,
        overlap_threshold=overlap_threshold,
    )


if __name__ == "__main__":
    # Example usage
    from pathlib import Path

    # Paths
    raw_data = Path("glassbox/data/raw/synthetic_notes.jsonl")
    redacted_data = Path("glassbox/data/processed/redacted_notes.jsonl")
    audit_log = Path("glassbox/data/processed/redaction_audit_log.jsonl")

    # Run pipeline
    print("Running redaction pipeline...")
    summary = run_redaction_pipeline(
        input_path=raw_data,
        output_path=redacted_data,
        audit_log_path=audit_log,
        confidence_threshold=0.5,
    )

    print("\nPipeline Summary:")
    print(f"  Documents processed: {summary['num_documents']}")
    print(f"  Entities detected: {summary['num_entities_detected']}")
    print(f"  Entities redacted: {summary['num_entities_redacted']}")
    print(f"  Output: {summary['output_path']}")
    print(f"  Audit log: {summary['audit_log_path']}")

    # Run evaluation
    print("\nRunning evaluation...")
    results = run_evaluation(
        dataset_path=raw_data,
        confidence_threshold=0.5,
        overlap_threshold=0.5,
    )

    print("\nEvaluation Results:")
    print(f"  Documents: {results['config']['num_documents']}")
    print(f"  Confidence threshold: {results['config']['confidence_threshold']}")
    print(f"  Overlap threshold: {results['config']['overlap_threshold']}")
    print("\nOverall Metrics:")
    print(f"  Precision: {results['overall']['precision']:.3f}")
    print(f"  Recall: {results['overall']['recall']:.3f}")
    print(f"  F1 Score: {results['overall']['f1']:.3f}")
    print("\nPer-Entity-Type Metrics:")
    for entity_type, metrics in sorted(results["per_type"].items()):
        print(f"  {entity_type}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")
        print(f"    TP/FP/FN: {metrics['tp']}/{metrics['fp']}/{metrics['fn']}")
