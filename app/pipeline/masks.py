"""Validate box-local COCO masks and map pixels into video coordinates."""


def _counts(encoded, total):
    if not isinstance(encoded, str) or not encoded:
        raise ValueError('invalid RLE')
    out, pos, running = [], 0, 0
    while pos < len(encoded):
        value = shift = 0
        while True:
            if pos >= len(encoded) or shift > 35:
                raise ValueError('invalid RLE')
            code = ord(encoded[pos]) - 48
            pos += 1
            if not 0 <= code <= 63:
                raise ValueError('invalid RLE')
            value |= (code & 31) << shift
            shift += 5
            if not (code & 32):
                if code & 16:
                    value |= -1 << shift
                break
        if len(out) > 2:
            value += out[-2]
        if value < 0:
            raise ValueError('invalid RLE run')
        out.append(value)
        running += value
        if running > total or len(out) > total + 1:
            raise ValueError('invalid RLE size')
    if running != total:
        raise ValueError('invalid RLE size')
    return out


def to_dedup_tracks(tracks, width, height):
    """Return masks as sets of (x,y). Missing confidence remains None.

    The dedup algorithm currently requires confidence; callers must handle
    missing confidence explicitly and must not fabricate a score.
    """
    if type(width) is not int or type(height) is not int or not (0 < width <= 4096 and 0 < height <= 4096):
        raise ValueError('invalid frame size')
    converted = []
    for track in tracks:
        new = {'track_id': track['track_id'], 'concept': track['concept'], 'frames': []}
        for frame in track['frames']:
            h, w = frame['mask_rle']['size']
            x, y = frame['mask_offset']
            if any(type(v) is not int for v in (h, w, x, y)) or h <= 0 or w <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError('mask out of frame bounds')
            runs = _counts(frame['mask_rle']['counts'], h * w)
            pixels, index = set(), 0
            for run_index, length in enumerate(runs):
                if run_index % 2:
                    pixels.update((x + i // h, y + i % h) for i in range(index, index + length))
                index += length
            new['frames'].append({'frame': frame['frame_index'], 'mask': pixels, 'score': frame['score']})
        converted.append(new)
    return converted
