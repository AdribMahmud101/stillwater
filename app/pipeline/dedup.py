"""Deduplicate overlapping cross-concept tracks using per-frame pixel masks.

Frame masks are sets of (x, y) pixels; provider-specific mask encodings must
be decoded by the caller. Frame numbers must use the same video timebase.
"""
from math import isfinite


def _mean(track):
    return sum(frame['score'] for frame in track['frames']) / len(track['frames'])


def _overlap(a, b, threshold):
    af = {f['frame']: f['mask'] for f in a['frames']}
    bf = {f['frame']: f['mask'] for f in b['frames']}
    common = af.keys() & bf.keys()
    if not common:
        return False
    matches = 0
    for number in common:
        first, second = af[number], bf[number]
        union = len(first | second)
        if union and len(first & second) / union >= threshold:
            matches += 1
    return matches * 2 >= len(common)


def deduplicate_tracks(tracks, min_track_frames=3, det_min=0.35, merge_iou=0.5):
    """Filter tracks and merge distinct concepts matching on half their shared frames.

    Representative is the highest mean-score track; aliases and source IDs retain
    attribution. Grouping requires every pair to match, avoiding transitive merges
    of separate nearby objects. Inputs are not mutated.
    """
    if min_track_frames < 1 or not 0 <= det_min <= 1 or not 0 <= merge_iou <= 1:
        raise ValueError('invalid tracking thresholds')
    valid = []
    for track in tracks:
        frames = track['frames']
        if len(frames) < min_track_frames:
            continue
        if len({frame['frame'] for frame in frames}) != len(frames):
            raise ValueError('duplicate frame number in track')
        if any(not isinstance(frame['mask'], (set, frozenset)) for frame in frames):
            raise ValueError('decode masks into pixel sets before deduplication')
        if any(not isinstance(frame['score'], (int, float)) or not isfinite(frame['score'])
               or not 0 <= frame['score'] <= 1 for frame in frames):
            raise ValueError('invalid detection score')
        if _mean(track) >= det_min:
            valid.append(track)
    groups = []
    for track in sorted(valid, key=_mean, reverse=True):
        for group in groups:
            if all(track['concept'] != other['concept'] and _overlap(track, other, merge_iou)
                   for other in group):
                group.append(track)
                break
        else:
            groups.append([track])
    result = []
    for group in groups:
        representative = dict(group[0])
        representative['aliases'] = sorted({t['concept'] for t in group[1:]})
        representative['source_track_ids'] = [t['track_id'] for t in group]
        result.append(representative)
    return result
