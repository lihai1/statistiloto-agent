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
