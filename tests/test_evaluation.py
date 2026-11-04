import unittest

from app.evaluation import EvaluationError, evaluate_clip


class EvaluationTests(unittest.TestCase):
    def test_per_class_mismatches_duplicates_and_worst(self):
        truth = [{'object_id': '1', 'type': 'bucket'}, {'object_id': '2', 'type': 'tire'}, {'object_id': '3', 'type': 'tire'}]
        predictions = [
            {'prediction_id': 'a', 'type': 'bucket', 'truth_id': '1'},
            {'prediction_id': 'b', 'type': 'bucket', 'truth_id': '1'},
            {'prediction_id': 'c', 'type': 'bottle', 'truth_id': '2'},
            {'prediction_id': 'd', 'type': 'bottle', 'truth_id': None},
        ]
        result = evaluate_clip(truth, predictions)
        self.assertEqual(result['classes']['bucket'], {'true_positive': 1, 'predicted': 2, 'truth': 1, 'precision': 0.5, 'recall': 1.0})
        self.assertEqual(result['classes']['tire']['recall'], 0.0)
        self.assertEqual(result['classes']['bottle']['precision'], 0.0)
        self.assertIsNone(result['classes']['bottle']['recall'])
        self.assertEqual(result['worst_class'], 'tire')
        self.assertEqual(result['absolute_count_error'], 1)
        self.assertAlmostEqual(result['duplicate_rate'], 1/3)

    def test_empty_data_has_undefined_metrics(self):
        result = evaluate_clip([], [])
        self.assertEqual(result['classes'], {})
        self.assertIsNone(result['worst_class'])
        self.assertIsNone(result['duplicate_rate'])
        self.assertEqual(result['absolute_count_error'], 0)

    def test_duplicate_predictions_do_not_change_match_based_on_order(self):
        truth = [{'object_id': '1', 'type': 'bucket'}]
        predictions = [{'prediction_id': 'wrong', 'type': 'bottle', 'truth_id': '1'}, {'prediction_id': 'right', 'type': 'bucket', 'truth_id': '1'}]
        a = evaluate_clip(truth, predictions)
        b = evaluate_clip(truth, list(reversed(predictions)))
        self.assertEqual(a, b)
        self.assertEqual(a['classes']['bucket']['recall'], 1)

    def test_invalid_data_rejected(self):
        for truth, predictions in [
            ([{'object_id':'a','type':'bucket'}, {'object_id':'a','type':'tire'}], []),
            ([], [{'prediction_id':'a','type':'bucket','truth_id':'missing'}]),
            ([], [{'prediction_id':'a','type':'bucket','truth_id':None}, {'prediction_id':'a','type':'bucket','truth_id':None}]),
            ([{'object_id':'','type':'bucket'}], []),
        ]:
            with self.subTest(truth=truth, predictions=predictions), self.assertRaises(EvaluationError):
                evaluate_clip(truth, predictions)

    def test_confusion_uses_one_match_per_truth_and_is_order_independent(self):
        from app.evaluation import risk_confusion
        truth = [{'object_id':'a','type':'bucket','risk_truth':'high'}, {'object_id':'b','type':'tire','risk_truth':'low'}]
        predictions = [
            {'prediction_id':'z','type':'bucket','truth_id':'a','risk':'medium'},
            {'prediction_id':'a','type':'bottle','truth_id':'a','risk':'low'},
            {'prediction_id':'b','type':'tire','truth_id':'b','risk':'review'},
            {'prediction_id':'x','type':'tire','truth_id':None,'risk':'high'},
        ]
        matrix = risk_confusion(truth, predictions)
        self.assertEqual(matrix, risk_confusion(truth, list(reversed(predictions))))
        self.assertEqual(matrix['matrix']['high']['medium'], 1)
        self.assertEqual(matrix['matrix']['low']['review'], 1)
        self.assertEqual(matrix['matched_pairs'], 2)
        self.assertEqual(matrix['unmatched_truth'], 0)
        self.assertEqual(matrix['unmatched_predictions'], 1)
        self.assertEqual(matrix['duplicate_predictions'], 1)

    def test_multi_clip_aggregation_sums_counts_without_merging_ids(self):
        from app.evaluation import evaluate_dataset
        clips = [
            {'clip_id':'a', 'truth':[{'object_id':'1','type':'bucket','risk_truth':'high'}],
             'predictions':[{'prediction_id':'p','type':'bucket','truth_id':'1','risk':'low'}]},
            {'clip_id':'b', 'truth':[{'object_id':'1','type':'bucket','risk_truth':'low'}, {'object_id':'2','type':'tire','risk_truth':'medium'}],
             'predictions':[{'prediction_id':'p','type':'bucket','truth_id':'1','risk':'low'}]},
        ]
        report=evaluate_dataset(clips)
        self.assertEqual(report['classes']['bucket']['truth'], 2)
        self.assertEqual(report['classes']['bucket']['recall'], 1.0)
        self.assertEqual(report['classes']['tire']['recall'], 0.0)
        self.assertEqual(report['worst_class'], 'tire')
        self.assertEqual(report['mean_absolute_count_error'], 0.5)
        self.assertEqual(report['risk_confusion']['matrix']['high']['low'], 1)
        self.assertEqual(report['risk_confusion']['unmatched_truth'], 1)
        with self.assertRaises(EvaluationError):
            evaluate_dataset([clips[0], clips[0]])

    def test_missing_risk_is_reported_not_assumed(self):
        from app.evaluation import risk_confusion
        truth = [{'object_id':'1', 'type':'bucket'}]
        prediction = [{'prediction_id':'1','type':'bucket','truth_id':'1','risk':'low'}]
        result = risk_confusion(truth, prediction)
        self.assertEqual(result['missing_risk_pairs'], 1)
        self.assertEqual(result['matched_pairs'], 0)
