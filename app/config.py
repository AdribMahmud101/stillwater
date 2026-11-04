"""Fail-closed public deployment configuration prerequisites.

This validates declared settings only. Runtime must also verify live clients,
privacy evidence, budget enforcement and readiness; configuration alone does
not make a deployment safe, and does not enable any endpoint.
"""
from dataclasses import dataclass
from decimal import Decimal


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class DeploymentConfig:
    daily_budget_usd: Decimal | None = Decimal('5.00')
    privacy_verified: bool = False
    provider_mode: str = 'disabled'
    segmenter_id: str = ''
    witness_a_id: str = ''
    witness_b_id: str = ''

    def validate_public(self):
        errors = []
        budget = self.daily_budget_usd
        if not isinstance(budget, Decimal) or not budget.is_finite() or budget <= 0:
            errors.append('a positive finite daily budget is required')
        if self.privacy_verified is not True:
            errors.append('privacy must be verified before public deployment')
        if self.provider_mode != 'live':
            errors.append('live provider mode is required')
        if self.segmenter_id != 'meta-sam-3.1':
            errors.append('configured live segmenter is not supported')
        if self.witness_a_id != 'baseten-spark' or self.witness_b_id != 'modal-glimmer':
            errors.append('independent live Spark and Glimmer witnesses are required')
        if errors:
            raise ConfigurationError('; '.join(errors))
