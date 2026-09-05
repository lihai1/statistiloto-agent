import datetime

from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Strength(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STRENGTH_UNSPECIFIED: _ClassVar[Strength]
    WEAK: _ClassVar[Strength]
    STRONG: _ClassVar[Strength]
STRENGTH_UNSPECIFIED: Strength
WEAK: Strength
STRONG: Strength

class DateWindow(_message.Message):
    __slots__ = ("to",)
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    to: _timestamp_pb2.Timestamp
    def __init__(self, to: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., **kwargs) -> None: ...

class GenerateFormRequest(_message.Message):
    __slots__ = ("how_many", "form_type", "will_be", "window", "strength")
    HOW_MANY_FIELD_NUMBER: _ClassVar[int]
    FORM_TYPE_FIELD_NUMBER: _ClassVar[int]
    WILL_BE_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    STRENGTH_FIELD_NUMBER: _ClassVar[int]
    how_many: int
    form_type: int
    will_be: _containers.RepeatedScalarFieldContainer[int]
    window: DateWindow
    strength: Strength
    def __init__(self, how_many: _Optional[int] = ..., form_type: _Optional[int] = ..., will_be: _Optional[_Iterable[int]] = ..., window: _Optional[_Union[DateWindow, _Mapping]] = ..., strength: _Optional[_Union[Strength, str]] = ...) -> None: ...

class NumberSet(_message.Message):
    __slots__ = ("numbers", "strong")
    NUMBERS_FIELD_NUMBER: _ClassVar[int]
    STRONG_FIELD_NUMBER: _ClassVar[int]
    numbers: _containers.RepeatedScalarFieldContainer[int]
    strong: int
    def __init__(self, numbers: _Optional[_Iterable[int]] = ..., strong: _Optional[int] = ...) -> None: ...

class GenerateFormResponse(_message.Message):
    __slots__ = ("forms",)
    FORMS_FIELD_NUMBER: _ClassVar[int]
    forms: _containers.RepeatedCompositeFieldContainer[NumberSet]
    def __init__(self, forms: _Optional[_Iterable[_Union[NumberSet, _Mapping]]] = ...) -> None: ...

class GetStatisticsRequest(_message.Message):
    __slots__ = ("how_many", "form_type", "window", "strength")
    HOW_MANY_FIELD_NUMBER: _ClassVar[int]
    FORM_TYPE_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    STRENGTH_FIELD_NUMBER: _ClassVar[int]
    how_many: int
    form_type: int
    window: DateWindow
    strength: Strength
    def __init__(self, how_many: _Optional[int] = ..., form_type: _Optional[int] = ..., window: _Optional[_Union[DateWindow, _Mapping]] = ..., strength: _Optional[_Union[Strength, str]] = ...) -> None: ...

class Pair(_message.Message):
    __slots__ = ("numbers", "count")
    NUMBERS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    numbers: _containers.RepeatedScalarFieldContainer[int]
    count: int
    def __init__(self, numbers: _Optional[_Iterable[int]] = ..., count: _Optional[int] = ...) -> None: ...

class GetStatisticsResponse(_message.Message):
    __slots__ = ("pairs",)
    PAIRS_FIELD_NUMBER: _ClassVar[int]
    pairs: _containers.RepeatedCompositeFieldContainer[Pair]
    def __init__(self, pairs: _Optional[_Iterable[_Union[Pair, _Mapping]]] = ...) -> None: ...

class AnalyzeRequest(_message.Message):
    __slots__ = ("form", "window")
    FORM_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    form: _containers.RepeatedScalarFieldContainer[int]
    window: DateWindow
    def __init__(self, form: _Optional[_Iterable[int]] = ..., window: _Optional[_Union[DateWindow, _Mapping]] = ...) -> None: ...

class AnalyzeResponse(_message.Message):
    __slots__ = ("frequency_groups", "archive_size")
    FREQUENCY_GROUPS_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_SIZE_FIELD_NUMBER: _ClassVar[int]
    frequency_groups: _containers.RepeatedCompositeFieldContainer[FrequencyGroup]
    archive_size: int
    def __init__(self, frequency_groups: _Optional[_Iterable[_Union[FrequencyGroup, _Mapping]]] = ..., archive_size: _Optional[int] = ...) -> None: ...

class FrequencyEntry(_message.Message):
    __slots__ = ("numbers", "count")
    NUMBERS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    numbers: _containers.RepeatedScalarFieldContainer[int]
    count: int
    def __init__(self, numbers: _Optional[_Iterable[int]] = ..., count: _Optional[int] = ...) -> None: ...

class FrequencyGroup(_message.Message):
    __slots__ = ("size", "combos", "entries")
    SIZE_FIELD_NUMBER: _ClassVar[int]
    COMBOS_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    size: int
    combos: int
    entries: _containers.RepeatedCompositeFieldContainer[FrequencyEntry]
    def __init__(self, size: _Optional[int] = ..., combos: _Optional[int] = ..., entries: _Optional[_Iterable[_Union[FrequencyEntry, _Mapping]]] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "version", "draws_loaded")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    DRAWS_LOADED_FIELD_NUMBER: _ClassVar[int]
    status: str
    version: str
    draws_loaded: int
    def __init__(self, status: _Optional[str] = ..., version: _Optional[str] = ..., draws_loaded: _Optional[int] = ...) -> None: ...

class SimulateRequest(_message.Message):
    __slots__ = ("form", "strong", "archive_window", "ticket_cost", "prize_amounts", "simulate_window")
    FORM_FIELD_NUMBER: _ClassVar[int]
    STRONG_FIELD_NUMBER: _ClassVar[int]
    ARCHIVE_WINDOW_FIELD_NUMBER: _ClassVar[int]
    TICKET_COST_FIELD_NUMBER: _ClassVar[int]
    PRIZE_AMOUNTS_FIELD_NUMBER: _ClassVar[int]
    SIMULATE_WINDOW_FIELD_NUMBER: _ClassVar[int]
    form: _containers.RepeatedScalarFieldContainer[int]
    strong: int
    archive_window: DateWindow
    ticket_cost: float
    prize_amounts: _containers.RepeatedScalarFieldContainer[float]
    simulate_window: DateWindow
    def __init__(self, form: _Optional[_Iterable[int]] = ..., strong: _Optional[int] = ..., archive_window: _Optional[_Union[DateWindow, _Mapping]] = ..., ticket_cost: _Optional[float] = ..., prize_amounts: _Optional[_Iterable[float]] = ..., simulate_window: _Optional[_Union[DateWindow, _Mapping]] = ...) -> None: ...

class SimulateTierHit(_message.Message):
    __slots__ = ("tier", "hits", "amount_per_hit", "total")
    TIER_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    AMOUNT_PER_HIT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    tier: int
    hits: int
    amount_per_hit: float
    total: float
    def __init__(self, tier: _Optional[int] = ..., hits: _Optional[int] = ..., amount_per_hit: _Optional[float] = ..., total: _Optional[float] = ...) -> None: ...

class SimulateDrawResult(_message.Message):
    __slots__ = ("draw_number", "draw_date", "winning_numbers", "winning_strong", "tier_hits", "prize_won", "ticket_cost", "used_real_prizes")
    DRAW_NUMBER_FIELD_NUMBER: _ClassVar[int]
    DRAW_DATE_FIELD_NUMBER: _ClassVar[int]
    WINNING_NUMBERS_FIELD_NUMBER: _ClassVar[int]
    WINNING_STRONG_FIELD_NUMBER: _ClassVar[int]
    TIER_HITS_FIELD_NUMBER: _ClassVar[int]
    PRIZE_WON_FIELD_NUMBER: _ClassVar[int]
    TICKET_COST_FIELD_NUMBER: _ClassVar[int]
    USED_REAL_PRIZES_FIELD_NUMBER: _ClassVar[int]
    draw_number: int
    draw_date: _timestamp_pb2.Timestamp
    winning_numbers: _containers.RepeatedScalarFieldContainer[int]
    winning_strong: int
    tier_hits: _containers.RepeatedCompositeFieldContainer[SimulateTierHit]
    prize_won: float
    ticket_cost: float
    used_real_prizes: bool
    def __init__(self, draw_number: _Optional[int] = ..., draw_date: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., winning_numbers: _Optional[_Iterable[int]] = ..., winning_strong: _Optional[int] = ..., tier_hits: _Optional[_Iterable[_Union[SimulateTierHit, _Mapping]]] = ..., prize_won: _Optional[float] = ..., ticket_cost: _Optional[float] = ..., used_real_prizes: _Optional[bool] = ...) -> None: ...

class SimulateTierSummary(_message.Message):
    __slots__ = ("tier", "label", "total_hits", "total_amount")
    TIER_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    TOTAL_HITS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_AMOUNT_FIELD_NUMBER: _ClassVar[int]
    tier: int
    label: str
    total_hits: int
    total_amount: float
    def __init__(self, tier: _Optional[int] = ..., label: _Optional[str] = ..., total_hits: _Optional[int] = ..., total_amount: _Optional[float] = ...) -> None: ...

class SimulateSummary(_message.Message):
    __slots__ = ("total_draws", "total_combinations", "total_spent", "total_won", "net", "tier_summaries", "draws_with_real_prizes")
    TOTAL_DRAWS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_COMBINATIONS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_SPENT_FIELD_NUMBER: _ClassVar[int]
    TOTAL_WON_FIELD_NUMBER: _ClassVar[int]
    NET_FIELD_NUMBER: _ClassVar[int]
    TIER_SUMMARIES_FIELD_NUMBER: _ClassVar[int]
    DRAWS_WITH_REAL_PRIZES_FIELD_NUMBER: _ClassVar[int]
    total_draws: int
    total_combinations: int
    total_spent: float
    total_won: float
    net: float
    tier_summaries: _containers.RepeatedCompositeFieldContainer[SimulateTierSummary]
    draws_with_real_prizes: int
    def __init__(self, total_draws: _Optional[int] = ..., total_combinations: _Optional[int] = ..., total_spent: _Optional[float] = ..., total_won: _Optional[float] = ..., net: _Optional[float] = ..., tier_summaries: _Optional[_Iterable[_Union[SimulateTierSummary, _Mapping]]] = ..., draws_with_real_prizes: _Optional[int] = ...) -> None: ...

class SimulateResponse(_message.Message):
    __slots__ = ("draws", "summary")
    DRAWS_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    draws: _containers.RepeatedCompositeFieldContainer[SimulateDrawResult]
    summary: SimulateSummary
    def __init__(self, draws: _Optional[_Iterable[_Union[SimulateDrawResult, _Mapping]]] = ..., summary: _Optional[_Union[SimulateSummary, _Mapping]] = ...) -> None: ...
