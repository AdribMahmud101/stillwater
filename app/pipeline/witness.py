"""Single Baseten witness. No risk classification or implied second witness."""
import base64
import json
import logging

from app.pipeline.budget_runner import BudgetAccountingError

_FIELDS = ('holds_water', 'covered', 'water_visible', 'exposed_to_rain')
_ALLOWED = frozenset(_FIELDS + ('orientation', 'object_label', 'evidence'))
_PROMPT = (
    'Describe the pictured container or water source using only visible evidence. Do not assign a risk level. '
    'Return only a JSON object with exactly these keys: holds_water, covered, water_visible, '
    'exposed_to_rain (each yes, no, or unsure); orientation (upright, inverted, on_side, '
    'or unsure); object_label (at most 3 words); evidence (at most 100 characters). '
    'Water guidance: dark or reflective liquid, a wet or shiny interior, floating debris or algae '
    'all count as water_visible=yes. holds_water=yes whenever the container currently contains liquid '
    'or visibly can hold it open to the sky and unobstructed; answer no only when clearly dry, covered, '
    'or inverted; answer unsure only when the image is too blurry or occluded to tell. '
    'The evidence text must be consistent with the fields: do not describe liquid or wetness while '
    'answering unsure or no for water_visible. '
    'Use unsure whenever the image does not establish an attribute.'
)


def _unknown():
    return {**{name: 'unsure' for name in _FIELDS}, 'orientation': 'unsure', 'object_label': '', 'evidence': ''}


def _validate(content):
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != _ALLOWED:
        raise ValueError('invalid fields')
    if any(not isinstance(parsed[key], str) or parsed[key] not in ('yes', 'no', 'unsure') for key in _FIELDS):
        raise ValueError('invalid attribute')
    if parsed['orientation'] not in ('upright', 'inverted', 'on_side', 'unsure'):
        raise ValueError('invalid orientation')
    if (not isinstance(parsed['object_label'], str) or len(parsed['object_label'].split()) > 3
            or not isinstance(parsed['evidence'], str) or len(parsed['evidence']) > 100):
        raise ValueError('invalid label or evidence')
    return parsed


def describe_container(client, image: bytes, mime_type: str) -> dict:
    """Describe one privacy-processed crop, retry malformed responses once.

    API failures and invalid responses yield unsure attributes. This is one
    witness only; callers must not treat it as agreement with another witness.
    """
    if not isinstance(image, bytes) or not image or mime_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise ValueError('valid image bytes and image MIME type required')
    url = 'data:' + mime_type + ';base64,' + base64.b64encode(image).decode('ascii')
    for attempt in range(2):
        prompt = _PROMPT + (' Your previous answer was invalid. Return a valid JSON object only.' if attempt else '')
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe this container or water source.'},
            {'type': 'image_url', 'image_url': {'url': url}},
        ]}]
        try:
            response = client.chat(messages, max_tokens=2048)
            content = response['choices'][0]['message']['content']
            if not isinstance(content, str):
                raise ValueError('missing content')  # noqa: TRY004
            return _validate(content)
        except BudgetAccountingError:
            raise
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError) as exc:
            logging.getLogger('stillwater.witness').warning(
                'witness call failed (attempt %d): %s: %s', attempt + 1, type(exc).__name__, str(exc)[:200])
            continue
    return _unknown()
