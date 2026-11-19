"""
Redaction pipeline - ties detection, redaction, and evaluation together.

Usage (from the repo root):
    python -m glassbox.src.redaction.pipeline --dataset realistic
    python -m glassbox.src.redaction.pipeline --dataset synthetic --threshold 0.3

Outputs (per dataset and confidence threshold):
    glassbox/data/processed/<dataset>/t<threshold>/redacted_notes.jsonl
    glassbox/data/processed/<dataset>/t<threshold>/redaction_audit_log.jsonl
    glassbox/results/redaction/<dataset>_t<threshold>.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .detector import PIIDetector
from .redactor import PIIRedactor
from .evaluate import RedactionEvaluator, evaluate_redaction

DATA_ROOT = Path("glassbox/data")
RESULTS_ROOT = Path("glassbox/results/redaction")


def raw_notes_path(dataset: str) -> Path:
    return DATA_ROOT / "raw" / dataset / "synthetic_notes.jsonl"


def processed_dir(dataset: str, threshold: float) -> Path:
    return DATA_ROOT / "processed" / dataset / f"t{threshold:.2f}"


def run_redaction_pipeline(
    input_path: Path,
    output_path: Path,
    audit_log_path: Optional[Path] = None,
    confidence_threshold: float = 0.5,
    overlap_threshold: float = 0.5,
    evaluate: bool = True,
) -> Dict[str, Any]:
    """
    Run the full redaction pipeline: detect → redact → save → (evaluate).

    Detection runs once per document; the evaluation scores exactly the spans
    that were redacted (after overlap resolution), so the reported metrics
    describe the sanitized output that downstream stages actually see.

    The redacted documents contain only sanitized text plus non-identifying
    metadata. The original text is deliberately NOT copied across: retrieval
    and generation read these files, and the raw notes stay in data/raw/.

    Returns:
        Summary statistics, plus evaluation results under "evaluation" if
        `evaluate` is True.
    """
    detector = PIIDetector(confidence_threshold=confidence_threshold)
    redactor = PIIRedactor()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_log_path:
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r") as f_in:
        documents = [json.loads(line) for line in f_in]

    total_entities_detected = 0
    redacted_spans = []
    audit_records = []

    with output_path.open("w") as f_out:
        for doc in documents:
            entities = detector.detect(doc["text"])
            total_entities_detected += len(entities)

            redacted_text, audit_log = redactor.redact(doc["text"], entities)
            redacted_spans.append(audit_log)

            redacted_doc = {
                "doc_id": doc["doc_id"],
                "text": redacted_text,
                "diagnosis": doc["diagnosis"],
                "note_type": doc.get("note_type", "note"),
                "medications": doc["medications"],
                "num_entities_redacted": len(audit_log),
            }
            f_out.write(json.dumps(redacted_doc) + "\n")

            for entry in audit_log:
                audit_records.append({"doc_id": doc["doc_id"], **entry})

    if audit_log_path:
        with audit_log_path.open("w") as f_audit:
            for record in audit_records:
                f_audit.write(json.dumps(record) + "\n")

    summary = {
        "num_documents": len(documents),
        "num_entities_detected": total_entities_detected,
        "num_entities_redacted": len(audit_records),
        "confidence_threshold": confidence_threshold,
        "output_path": str(output_path),
        "audit_log_path": str(audit_log_path) if audit_log_path else None,
    }

    if evaluate:
        evaluator = RedactionEvaluator(detector=detector, overlap_threshold=overlap_threshold)
        summary["evaluation"] = evaluator.score(documents, redacted_spans)

    return summary


def run_evaluation(
    dataset_path: Path,
    confidence_threshold: float = 0.5,
    overlap_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate raw detector output (before overlap resolution) on a dataset."""
    return evaluate_redaction(
        dataset_path=dataset_path,
        confidence_threshold=confidence_threshold,
        overlap_threshold=overlap_threshold,
    )


def print_results(summary: Dict[str, Any]) -> None:
    print("\nPipeline Summary:")
    print(f"  Documents processed: {summary['num_documents']}")
    print(f"  Entities detected: {summary['num_entities_detected']}")
    print(f"  Entities redacted: {summary['num_entities_redacted']}")
    print(f"  Output: {summary['output_path']}")
    print(f"  Audit log: {summary['audit_log_path']}")

    results = summary.get("evaluation")
    if not results:
        return
    o = results["overall"]
    print(f"\nEvaluation (confidence {summary['confidence_threshold']}, IoU {results['config']['overlap_threshold']}):")
    print(f"  Overall  P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}  (TP {o['tp']} FP {o['fp']} FN {o['fn']})")
    for entity_type, m in sorted(results["per_type"].items()):
        print(f"  {entity_type:<14} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
              f"(TP {m['tp']} FP {m['fp']} FN {m['fn']})")

    masking = results["masking"]
    print("\nMasking (type-agnostic privacy view):")
    for name, m in [("Overall", masking["overall"]), *sorted(masking["per_type"].items())]:
        print(f"  {name:<14} masked {m['masked_recall']:.3f}  fully masked {m['fully_masked_rate']:.3f}  "
              f"char leakage {m['char_leakage']:.3f}  (n={m['n']})")


def main():
    parser = argparse.ArgumentParser(description="Run GlassBox PII redaction on a dataset")
    parser.add_argument("--dataset", choices=["synthetic", "realistic"], default="realistic")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detector confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for evaluation matching")
    args = parser.parse_args()

    out_dir = processed_dir(args.dataset, args.threshold)
    print(f"Running redaction on {args.dataset} (threshold {args.threshold})...")
    summary = run_redaction_pipeline(
        input_path=raw_notes_path(args.dataset),
        output_path=out_dir / "redacted_notes.jsonl",
        audit_log_path=out_dir / "redaction_audit_log.jsonl",
        confidence_threshold=args.threshold,
        overlap_threshold=args.iou,
    )
    print_results(summary)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_ROOT / f"{args.dataset}_t{args.threshold:.2f}.json"
    with results_path.open("w") as f:
        json.dump({"dataset": args.dataset, **summary}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
