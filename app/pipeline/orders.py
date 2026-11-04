"""Deterministic bilingual cleanup orders based on the published action table.

Bangla copy is a draft pending native-speaker review. Dates and assignees are
never inferred here; callers may only pass validated, explicitly supplied data.
"""
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class WorkOrder:
    container_ids: tuple[str, ...]
    action_key: str
    action_en: str
    action_bn: str
    priority: int
    due_date: date | None = None
    assignee: str | None = None
    status: str = 'open'


_ACTIONS = {
    'high': ('empty_scrub', 'Empty and scrub today.', 'আজই খালি করে ঘষে পরিষ্কার করুন।', 1),
    'review': ('inspect', 'Inspector to check on site.', 'পরিদর্শক সরেজমিনে দেখবেন।', 2),
    'medium': ('cover_or_invert', 'Cover tightly or turn upside down.', 'শক্ত করে ঢেকে দিন বা উল্টে রাখুন।', 3),
    'low': ('recheck', 'No action now, recheck in 7 days.', 'এখন কিছু করার নেই, ৭ দিন পর আবার দেখুন।', 4),
}


def create_work_order(container_id: str, risk: str, *, object_label: str = '',
                      due_date: date | None = None, assignee: str | None = None) -> WorkOrder:
    if not isinstance(container_id, str) or not container_id.strip():
        raise ValueError('A container ID is required.')
    if not isinstance(risk, str) or risk not in _ACTIONS:
        raise ValueError('Unknown risk level.')
    if due_date is not None and (not isinstance(due_date, date) or isinstance(due_date, datetime)):
        raise ValueError('Due date must be a date.')
    if assignee is not None and (not isinstance(assignee, str) or not assignee.strip()):
        raise ValueError('Assignee must be a nonempty string.')
    if not isinstance(object_label, str):
        raise ValueError('Object label must be a string.')  # noqa: TRY004
    key, en, bn, priority = _ACTIONS[risk]
    if risk == 'high' and object_label.strip().lower() in ('tire', 'tyre'):
        en += ' Tyres: remove, or drill a drain hole.'
        bn += ' টায়ার: সরিয়ে ফেলুন বা পানি বের হওয়ার ছিদ্র করুন।'
        key = 'empty_scrub_tire'
    return WorkOrder((container_id.strip(),), key, en, bn, priority, due_date, assignee.strip() if assignee else None)
