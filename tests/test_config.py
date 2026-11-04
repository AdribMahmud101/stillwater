import unittest
from decimal import Decimal
from app.config import DeploymentConfig, ConfigurationError


def ready(**changes):
    data = dict(daily_budget_usd=Decimal('5'), privacy_verified=True,
                provider_mode='live', segmenter_id='meta-sam-3.1',
                witness_a_id='baseten-spark', witness_b_id='modal-glimmer')
    data.update(changes)
    return DeploymentConfig(**data)


class ConfigTests(unittest.TestCase):
    def test_valid_public_config(self):
        ready().validate_public()

    def test_invalid_budget_fails_closed(self):
        for value in (None, Decimal('0'), Decimal('-1'), Decimal('NaN'), Decimal('Infinity'), '5', True):
            with self.subTest(value=value), self.assertRaises(ConfigurationError): ready(daily_budget_usd=value).validate_public()

    def test_unverified_privacy_and_fakes_rejected(self):
        for changes in ({'privacy_verified': False}, {'privacy_verified': 1}, {'provider_mode': 'mock'},
                        {'segmenter_id': ''}, {'witness_a_id': 'spark', 'witness_b_id': 'SPARK'}):
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError): ready(**changes).validate_public()

    def test_absent_or_fake_witnesses_rejected(self):
        for id in ('', None, 'mock', 'fake-provider', 'stub'):
            with self.subTest(id=id), self.assertRaises(ConfigurationError): ready(witness_b_id=id).validate_public()
