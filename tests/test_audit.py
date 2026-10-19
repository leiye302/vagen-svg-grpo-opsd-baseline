import copy
import unittest
from svg_opsd.audit import validation_report


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.ids = [f'dev_{i:03}' for i in range(64)]
        self.manifest = {'splits': {'dev': {'task_ids': self.ids, 'sha256': 'frozen'}}}
        self.rows = [{'task_id': task, 'config_id': task, 'output_str': 'sampled text',
                     'metrics': {'score': i/64, 'dino': .9}} for i, task in enumerate(self.ids)]

    def test_fixed_ids_order_independent_mean(self):
        report = validation_report(list(reversed(self.rows)), self.manifest, 10, 'base')
        self.assertEqual(report['task_ids'], self.ids)
        self.assertAlmostEqual(report['means']['val/dev64/score'], 31.5/64)
        self.assertFalse(report['test_used'])

    def test_duplicate_identity_rejected(self):
        self.rows[-1]['task_id'] = self.rows[0]['task_id']
        with self.assertRaises(RuntimeError):
            validation_report(self.rows, self.manifest, 10, 'base')

    def test_changed_or_missing_identity_rejected(self):
        self.rows[-1]['task_id'] = 'test_task'
        with self.assertRaises(RuntimeError):
            validation_report(self.rows, self.manifest, 10, 'base')
        with self.assertRaises(RuntimeError):
            validation_report(self.rows[:-1], self.manifest, 10, 'base')

    def test_nonfinite_metric_rejected(self):
        self.rows[-1]['metrics']['score'] = float('nan')
        with self.assertRaises(RuntimeError):
            validation_report(self.rows, self.manifest, 10, 'base')


if __name__ == '__main__':
    unittest.main()
