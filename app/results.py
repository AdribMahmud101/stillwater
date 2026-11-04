"""Private, atomic persistence for results derived from privacy-filtered media.

Privacy verification is a caller attestation, not independently detectable from
pixels. Never pass source uploads to this store. Video is not stored here.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import uuid
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.evidence import Evidence
from app.pipeline.features import extract_match_features
from app.pipeline.orders import WorkOrder, create_work_order


class ResultsError(RuntimeError):
    pass


def _scan_id(value: str) -> str:
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError
    except (TypeError, ValueError, AttributeError) as exc:
        raise ResultsError('Invalid scan ID.') from exc
    return value


def _decode(data: bytes, mode: int):
    if not isinstance(data, bytes) or not data or len(data) > 20 * 1024 * 1024:
        raise ResultsError('Invalid evidence image.')
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), mode)
    if decoded is None or decoded.size == 0:
        raise ResultsError('Invalid evidence image.')
    return decoded


def _validate_cost(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise ResultsError('Invalid scan cost.')
    try:
        finite = value.is_finite()
    except (AttributeError, ValueError):
        raise ResultsError('Invalid scan cost.')
    if not finite or value < 0:
        raise ResultsError('Invalid scan cost.')
    # Persist as a decimal STRING in the manifest.
    return str(value)


def _validate_timings(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ResultsError('Invalid scan timings.')
    cleaned: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or len(key) > 120:
            raise ResultsError('Invalid scan timings.')
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ResultsError('Invalid scan timings.')
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ResultsError('Invalid scan timings.')
        cleaned[key] = number
    return cleaned


def _validate_checks(checks):
    fields = {'holds_water', 'covered', 'water_visible', 'orientation', 'exposed_to_rain'}
    if not isinstance(checks, dict) or set(checks) != {'agree', 'a', 'b'} or type(checks['agree']) is not bool:
        raise ResultsError('Invalid checks.')
    for key in ('a', 'b'):
        answer = checks[key]
        if not isinstance(answer, dict) or set(answer) != fields | {'label'}:
            raise ResultsError('Invalid checks.')
        label = answer['label']
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise ResultsError('Invalid checks.')
        for field in fields:
            value = answer[field]
            allowed = ('upright', 'inverted', 'on_side', 'unsure') if field == 'orientation' else ('yes', 'no', 'unsure')
            if value is not None and (not isinstance(value, str) or value not in allowed):
                raise ResultsError('Invalid checks.')
    return checks


class ResultsStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)

    def save(self, scan_id: str, containers: list[dict], *, privacy_verified: bool, annotated_video: bytes | None = None, timeline: list[dict] | None = None, cost_usd: Decimal | None = None, timings: dict | None = None, media_type: str | None = None) -> dict:
        scan_id = _scan_id(scan_id)
        if privacy_verified is not True:
            raise ResultsError('Privacy must be verified before persisting evidence.')
        if not isinstance(containers, list):
            raise ResultsError('Containers must be a list.')
        if media_type is None:
            media_type = 'video'
        if media_type not in ('image', 'video'):
            raise ResultsError('Invalid media type.')
        cost_text = _validate_cost(cost_usd)
        timings_doc = _validate_timings(timings)
        dest = self.root / scan_id
        if dest.exists():
            raise ResultsError('Results already exist.')
        tmp = Path(tempfile.mkdtemp(prefix='.result-', dir=self.root))
        try:
            rows = []
            feature_inputs = []
            ids = set()
            for index, item in enumerate(containers):
                if not isinstance(item, dict) or not {'evidence', 'risk', 'rule_id'} <= set(item) or set(item) - {'evidence', 'risk', 'rule_id', 'work_order', 'concept', 'checks'}:
                    raise ResultsError('Invalid container result.')
                evidence, risk, rule = item['evidence'], item['risk'], item['rule_id']
                if not isinstance(evidence, Evidence) or not isinstance(evidence.track_id, str) or not evidence.track_id or evidence.track_id in ids or risk not in {'high', 'medium', 'low', 'review'} or not isinstance(rule, str) or not re.fullmatch(r'R[0-9]+[a-z]?', rule):
                    raise ResultsError('Invalid container metadata.')
                ids.add(evidence.track_id)
                if type(evidence.frame_index) is not int or evidence.frame_index < 0 or not isinstance(evidence.timestamp_seconds, (int, float)) or not math.isfinite(evidence.timestamp_seconds) or evidence.timestamp_seconds < 0:
                    raise ResultsError('Invalid evidence timestamp.')
                frame = _decode(evidence.frame_jpeg, cv2.IMREAD_COLOR)
                crop = _decode(evidence.crop_jpeg, cv2.IMREAD_COLOR)
                mask = _decode(evidence.mask_png, cv2.IMREAD_GRAYSCALE)
                box = evidence.crop_box
                if not isinstance(box, tuple) or len(box) != 4 or any(type(v) is not int for v in box):
                    raise ResultsError('Invalid crop box.')
                x1, y1, x2, y2 = box
                height, width = frame.shape[:2]
                if mask.shape != (height, width) or not np.isin(mask, [0, 255]).all() or not mask.any() or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) or crop.shape[:2] != (y2-y1, x2-x1):
                    raise ResultsError('Evidence dimensions or mask are invalid.')
                concept = item.get('concept')
                if concept is not None and (not isinstance(concept, str) or not concept.strip() or len(concept) > 120):
                    raise ResultsError('Invalid object concept.')
                mask_crop = mask[y1:y2, x1:x2].astype(bool)
                if not mask_crop.any() or np.count_nonzero(mask_crop) != np.count_nonzero(mask):
                    raise ResultsError('Evidence mask falls outside crop.')
                feature_inputs.append((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), mask_crop, evidence.frame_index, evidence.timestamp_seconds, concept))
                folder = f'{index:06d}'
                (tmp / folder).mkdir(mode=0o700)
                for name, data in [('frame.jpg', evidence.frame_jpeg), ('crop.jpg', evidence.crop_jpeg), ('mask.png', evidence.mask_png)]:
                    path = tmp / folder / name
                    with path.open('xb') as f:
                        os.chmod(path, 0o600)
                        f.write(data)
                order = item.get('work_order')
                order_data = None
                if order is not None:
                    if not isinstance(order, WorkOrder):
                        raise ResultsError('Invalid work order.')
                    label = 'tire' if order.action_key == 'empty_scrub_tire' else ''
                    try:
                        expected = create_work_order(evidence.track_id, risk, object_label=label,
                                                     due_date=order.due_date, assignee=order.assignee)
                    except (ValueError, TypeError) as exc:
                        raise ResultsError('Invalid work order.') from exc
                    if order != expected:
                        raise ResultsError('Work order does not match container or risk.')
                    order_data = {'container_ids': list(order.container_ids), 'action_key': order.action_key,
                                  'action_en': order.action_en, 'action_bn': order.action_bn,
                                  'priority': order.priority,
                                  'due_date': order.due_date.isoformat() if order.due_date else None,
                                  'assignee': order.assignee, 'status': order.status}
                checks = _validate_checks(item['checks']) if 'checks' in item else None
                rows.append({'work_order': order_data, 'track_id': evidence.track_id, 'risk': risk, 'rule_id': rule,
                             'frame_index': evidence.frame_index, 'timestamp_seconds': evidence.timestamp_seconds,
                             'crop_box': list(box), 'frame': f'{folder}/frame.jpg', 'crop': f'{folder}/crop.jpg', 'mask': f'{folder}/mask.png'})
                if checks is not None:
                    rows[-1]['checks'] = checks
            doc = {'scan_id': scan_id, 'privacy_verified': True, 'containers': rows, 'media_type': media_type}
            if cost_text is not None:
                doc['cost_usd'] = cost_text
            if timings_doc is not None:
                doc['timings'] = timings_doc
            if annotated_video is None:
                if timeline is not None:
                    raise ResultsError('Timeline requires an annotated video.')
                if media_type == 'image':
                    # Image scans without containers may have no annotated media.
                    doc['timeline'] = None
            else:
                if media_type == 'image':
                    if not isinstance(annotated_video, bytes) or not 0 < len(annotated_video) <= 20 * 1024 * 1024 or timeline is not None:
                        raise ResultsError('Invalid annotated image or timeline.')
                    decoded = cv2.imdecode(np.frombuffer(annotated_video, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if decoded is None or decoded.size == 0:
                        raise ResultsError('Invalid annotated image.')
                    image_path = tmp / 'annotated.jpg'
                    with image_path.open('xb') as f:
                        os.chmod(image_path, 0o600)
                        f.write(annotated_video)
                    for row, (crop_rgb, mask_crop, evidence_index, evidence_time, concept) in zip(rows, feature_inputs, strict=True):
                        if evidence_index != 0 or abs(float(evidence_time) - 0.0) > 0.001:
                            raise ResultsError('Evidence timestamp does not match image.')
                        if concept is not None:
                            row['concept'] = concept.strip()
                            row.update(extract_match_features(crop_rgb, evidence_time, 1.0, mask_crop))
                    doc['annotated_video'] = 'annotated.jpg'
                    doc['timeline'] = None
                else:
                    if not isinstance(annotated_video, bytes) or not 0 < len(annotated_video) <= 200 * 1024 * 1024 or not isinstance(timeline, list):
                        raise ResultsError('Invalid annotated video or timeline.')
                    video_path = tmp / 'annotated.mp4'
                    with video_path.open('xb') as f:
                        os.chmod(video_path, 0o600)
                        f.write(annotated_video)
                    cap = cv2.VideoCapture(str(video_path))
                    try:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
                            raise ResultsError('Invalid annotated video.')
                        frame_count = 0
                        while cap.read()[0]:
                            frame_count += 1
                        if frame_count == 0:
                            raise ResultsError('Annotated video contains no frames.')
                    finally:
                        cap.release()
                    seen_tracks = set()
                    risks = {row['track_id']: row['risk'] for row in rows}
                    for entry in timeline:
                        if not isinstance(entry, dict) or set(entry) != {'track_id', 'risk', 'frame_index', 'timestamp_seconds'}:
                            raise ResultsError('Invalid timeline entry.')
                        ident, risk, index, stamp = entry['track_id'], entry['risk'], entry['frame_index'], entry['timestamp_seconds']
                        if not isinstance(ident, str) or ident not in risks or ident in seen_tracks or risk != risks[ident] or type(index) is not int or not 0 <= index < frame_count or isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or abs(stamp - index / fps) > 0.001:
                            raise ResultsError('Timeline does not match video and container records.')
                        seen_tracks.add(ident)
                    if seen_tracks != risks.keys():
                        raise ResultsError('Timeline is missing a container.')
                    for row, (crop_rgb, mask_crop, evidence_index, evidence_time, concept) in zip(rows, feature_inputs, strict=True):
                        if evidence_index >= frame_count or abs(evidence_time - evidence_index / fps) > .001:
                            raise ResultsError('Evidence timestamp does not match video.')
                        if concept is not None:
                            row['concept'] = concept.strip()
                            row.update(extract_match_features(crop_rgb, evidence_time, frame_count / fps, mask_crop))
                    doc['annotated_video'] = 'annotated.mp4'
                    doc['timeline'] = timeline
            manifest = tmp / 'results.json'
            with manifest.open('x', encoding='utf-8') as f:
                os.chmod(manifest, 0o600)
                json.dump(doc, f, ensure_ascii=False, allow_nan=False)
            # A populated existing result directory cannot be replaced by rename.
            os.rename(tmp, dest)
            return doc
        except ResultsError:
            raise
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            raise ResultsError('Could not save scan results.') from exc
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def load(self, scan_id: str) -> dict:
        scan_id = _scan_id(scan_id)
        try:
            with (self.root / scan_id / 'results.json').open(encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError) as exc:
            raise ResultsError('Scan results are unavailable.') from exc
