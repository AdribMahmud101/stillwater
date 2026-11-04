"""Evaluate human-labeled clips; never runs models or creates ground truth.

Example:
    python scripts/eval.py --clip roof1 data/labels/roof1.csv data/labels/roof1_predictions.json --output data/eval/results.json

Truth CSV requires object_id,type and optionally risk_truth. Predictions JSON is
an array of objects with prediction_id,type,truth_id and optionally risk. Each
truth_id is a human-selected object_id or null for a false positive. Missing
risk labels are excluded from the confusion matrix and explicitly counted.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import EvaluationError, evaluate_dataset


def read_clip(clip_id: str, truth_path: str, predictions_path: str) -> dict:
    with Path(truth_path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {'object_id', 'type'} <= set(reader.fieldnames):
            raise EvaluationError('truth CSV needs object_id and type columns')
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise EvaluationError('truth CSV has duplicate columns')
        truth = list(reader)
    with Path(predictions_path).open(encoding='utf-8-sig') as stream:
        predictions = json.load(stream)
    if not isinstance(predictions, list):
        raise EvaluationError('predictions JSON must be an array of human-matched rows')
    return {'clip_id': clip_id, 'truth': truth, 'predictions': predictions}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Evaluate manually labeled and matched StillWater predictions.')
    parser.add_argument('--clip', action='append', nargs=3, metavar=('CLIP_ID', 'TRUTH_CSV', 'MATCHED_JSON'), required=True)
    parser.add_argument('--output', type=Path, help='Write JSON report atomically; defaults to stdout')
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() in {Path(path).resolve() for _, truth, pred in args.clip for path in (truth, pred)}:
            raise EvaluationError('output cannot overwrite an input file')
        report = evaluate_dataset([read_clip(*clip) for clip in args.clip])
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.eval-', dir=args.output.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as out:
                    out.write(serialized)
                os.replace(temporary, args.output)
            finally:
                Path(temporary).unlink(missing_ok=True)
        else:
            sys.stdout.write(serialized)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f'evaluation failed: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
