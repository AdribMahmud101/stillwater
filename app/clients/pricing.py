"""Exact cost arithmetic; prices must be supplied from current provider rates."""
from decimal import Decimal


def unit_cost(count: int, rate: Decimal, per: int) -> Decimal:
    if type(count) is not int or count < 0 or type(per) is not int or per <= 0:
        raise ValueError('invalid usage')
    if not isinstance(rate, Decimal) or not rate.is_finite() or rate < 0:
        raise ValueError('invalid rate')
    return Decimal(count) * rate / Decimal(per)


def token_cost(input_tokens: int, output_tokens: int, input_rate: Decimal, output_rate: Decimal) -> Decimal:
    return unit_cost(input_tokens, input_rate, 1_000_000) + unit_cost(output_tokens, output_rate, 1_000_000)
