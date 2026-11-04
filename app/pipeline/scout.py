"""Validate and bound scene-adaptive segmentation prompts."""
import json
from collections.abc import Mapping

BASE_CONCEPTS = ('tire', 'bucket', 'flower pot', 'plastic container', 'plastic drum', 'puddle', 'standing water', 'drain')
# Explicit vocabulary prevents arbitrary model output becoming a segmentation prompt.
# Includes discrete containers plus ground / structural standing-water sources
# (core mosquito breeding sites) so SAM is prompted for puddles, drains, etc.
CONTAINER_NOUNS = frozenset((*BASE_CONCEPTS, 'bottle', 'coconut shell', 'water tank', 'tarp', 'plant saucer',
    'wheelbarrow', 'basin', 'bowl', 'can', 'tin can', 'barrel', 'drum', 'pot', 'plant pot',
    'watering can', 'jug', 'pitcher', 'cup', 'mug', 'tray', 'tub', 'tank', 'jar', 'vase',
    'dish', 'saucer', 'cooler', 'box', 'crate', 'plastic tub', 'plastic bowl', 'plastic bucket',
    'metal bucket', 'paint can', 'discarded tire', 'plastic bottle', 'storage bin', 'tarpaulin',
    'water pool', 'pond', 'ditch', 'clogged drain', 'gutter', 'water-filled hole', 'wet roof',
    'waterlogged area'))
SETTINGS = frozenset(('rooftop', 'lane', 'yard', 'construction', 'indoor', 'other'))

# Total concepts sent to SAM must stay <= 10 (app/pipeline/sam_track.py enforces 10).
MAX_CONCEPTS = 10
MAX_EXTRAS = MAX_CONCEPTS - len(BASE_CONCEPTS)


def validate_scout(raw):
    """Return validated scene and filtered extras, or ValueError for retry.

    Unknown nouns are ignored. Structurally malformed responses are rejected;
    callers may retry once and then use scout_concepts(None) for the base list.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            raise ValueError('invalid JSON') from exc
    if not isinstance(raw, Mapping) or set(raw) != {'scene', 'extra_concepts'}:
        raise ValueError('expected scene and extra_concepts')
    scene, extras = raw['scene'], raw['extra_concepts']
    if not isinstance(scene, Mapping) or set(scene) != {'setting', 'notes'}:
        raise ValueError('invalid scene')
    if not isinstance(scene['setting'], str) or scene['setting'] not in SETTINGS or not isinstance(scene['notes'], str) or len(scene['notes']) > 120:
        raise ValueError('invalid scene setting or notes')
    if not isinstance(extras, list):
        raise ValueError('extra_concepts must be a list')  # noqa: TRY004
    accepted = []
    for item in extras:
        if not isinstance(item, Mapping) or set(item) != {'noun', 'why'} or not isinstance(item['noun'], str) or not isinstance(item['why'], str) or len(item['why']) > 80:
            raise ValueError('invalid concept')
        noun = ' '.join(item['noun'].lower().split())
        if len(noun.split()) > 3:
            raise ValueError('concept exceeds three words')
        if noun in CONTAINER_NOUNS and noun not in BASE_CONCEPTS and noun not in accepted and len(accepted) < MAX_EXTRAS:
            accepted.append(noun)
    return {'scene': dict(scene), 'extra_concepts': accepted}


def scout_concepts(raw):
    """Return safe base vocabulary plus up to MAX_EXTRAS validated concepts."""
    try:
        result = validate_scout(raw)
    except ValueError:
        return list(BASE_CONCEPTS)
    return [*BASE_CONCEPTS, *result['extra_concepts']]
