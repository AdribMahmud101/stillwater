"""Scene scout using already privacy-processed keyframes."""
import base64
import json

from app.pipeline.budget_runner import BudgetAccountingError

from app.pipeline.scout import BASE_CONCEPTS, validate_scout

_PROMPT = (
    'Inspect the scene for water sources and containers that can hold rainwater, '
    'including ground and structural standing-water sources such as puddles, '
    'standing water, drains, gutters, ditches, ponds, and waterlogged areas, as well as '
    'discrete containers. Return only a JSON object with '
    'scene: {setting: rooftop|lane|yard|construction|indoor|other, notes: string at most 120 characters} '
    'and extra_concepts: array of at most four objects {noun: at most 3 words, why: at most 80 characters}. '
    'Suggest concrete water-source or container nouns visible in the scene in addition to: ' + ', '.join(BASE_CONCEPTS) + '.'
)


def scout_scene(client, keyframes):
    """Return validated raw scene JSON, or None for base-concept fallback.

    Keyframes must already be privacy-processed. Caller must reserve budget.
    Provider JSON mode is deliberately not assumed. Invalid output is retried
    once using a distinct request; each model response uses the provider cache.
    """
    if not keyframes:
        return None
    if len(keyframes) > 8:
        raise ValueError('at most eight keyframes are supported')
    images = []
    for frame in keyframes:
        image = frame.jpeg_bytes
        if not isinstance(image, bytes) or not image:
            raise ValueError('keyframe JPEG is required')
        images.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image).decode('ascii')}})
    for attempt in range(2):
        prompt = _PROMPT + (' Your last response was invalid; return valid JSON only.' if attempt else '')
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': [
            {'type': 'text', 'text': 'Identify additional water-source and container nouns visible in the scene.'}, *images]}]
        try:
            response = client.chat(messages, max_tokens=2048)
            text = response['choices'][0]['message']['content']
            if not isinstance(text, str):
                raise ValueError('response has no text')
            raw = json.loads(text)
            validate_scout(raw)
            return raw
        except BudgetAccountingError:
            raise
        except (RuntimeError, ValueError, TypeError, KeyError, IndexError):
            continue
    return None
