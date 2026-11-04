"""Convert completed SAM video text using Meta's official parser.

Masks are box-local COCO RLE. Their offset must be applied when rendering.
Track IDs are only meaningful within one request and may change after occlusion.
"""
from meta_sam_parser import (
    CompletedOutcome,
    decode_mask_to_rle,
    video_segmentation_format,
)


class SAMParseError(ValueError):
    pass


def parse_sam_video(raw: str, concept: str) -> list[dict]:
    if not isinstance(raw, str) or not isinstance(concept, str) or not concept.strip():
        raise ValueError('raw output and concept are required')
    parser = video_segmentation_format().create_parser()
    try:
        parser.push(raw)
        result = parser.finish(CompletedOutcome()).result
        if result.diagnostics or any(record.kind == 'text' for record in result.records):
            raise SAMParseError('SAM output contains malformed or unrecognized records')
        boxes, masks = {}, {}
        for record in result.records:
            if record.frame is None:
                raise SAMParseError('video record has no frame index')
            key = (record.object_id, record.frame.frame_index)
            if record.kind == 'box':
                if key in boxes:
                    raise SAMParseError('duplicate box')
                boxes[key] = record
            elif record.kind == 'mask':
                if key in masks:
                    raise SAMParseError('duplicate mask')
                masks[key] = record
        if set(boxes) != set(masks):
            raise SAMParseError('box and mask records do not match')
        tracks = {}
        for key in sorted(boxes, key=lambda k: (k[0], k[1])):
            box, mask = boxes[key], masks[key]
            rle = decode_mask_to_rle(mask.mask)
            bounds = mask.bounds
            frame = {'frame_index': key[1], 'box': [box.left, box.top, box.right, box.bottom],
                     'mask_rle': {'size': list(rle.size), 'counts': rle.counts},
                     'mask_offset': [bounds.left, bounds.top], 'score': box.confidence}
            tracks.setdefault(key[0], {'track_id': key[0], 'concept': concept, 'frames': []})['frames'].append(frame)
        return list(tracks.values())
    except SAMParseError:
        raise
    except (ValueError, TypeError, RuntimeError) as exc:
        raise SAMParseError('SAM output could not be decoded') from exc
