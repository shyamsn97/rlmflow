"""Typed JSON-line protocol models for minimal remote REPLs."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseRequest(WireModel):
    id: str
    cmd: str
    repl_id: str = "root"


class PingRequest(BaseRequest):
    cmd: Literal["ping"] = "ping"


class CapabilitiesRequest(BaseRequest):
    cmd: Literal["capabilities"] = "capabilities"


class RunRequest(BaseRequest):
    cmd: Literal["run"] = "run"
    code: str


class InjectRequest(BaseRequest):
    cmd: Literal["inject"] = "inject"
    name: str
    value: Any


class RemoveRequest(BaseRequest):
    cmd: Literal["remove"] = "remove"
    name: str


class SetEnvRequest(BaseRequest):
    cmd: Literal["set_env"] = "set_env"
    values: dict[str, Any]


class InjectProxyRequest(BaseRequest):
    cmd: Literal["inject_proxy"] = "inject_proxy"
    name: str
    is_async: bool = False


class InjectImportRequest(BaseRequest):
    cmd: Literal["inject_import"] = "inject_import"
    name: str
    module: str
    qualname: str


class InjectSourceRequest(BaseRequest):
    cmd: Literal["inject_source"] = "inject_source"
    name: str
    func_name: str
    source: str


ReplRequest = Annotated[
    PingRequest
    | CapabilitiesRequest
    | RunRequest
    | InjectRequest
    | RemoveRequest
    | SetEnvRequest
    | InjectProxyRequest
    | InjectImportRequest
    | InjectSourceRequest,
    Field(discriminator="cmd"),
]


class CapabilityMap(WireModel):
    namespace_snapshot: bool = False
    filesystem_snapshot: bool = False
    process_checkpoint: bool = False
    fork: Literal["none", "replay", "snapshot", "process"] = "none"


class ReplResponse(WireModel):
    id: str
    ok: bool = True
    output: str | None = None
    errored: bool = False
    error: str | None = None
    value: Any = None
    done: bool = False
    env: dict[str, Any] | None = None
    capabilities: CapabilityMap | None = None


class ProxyCall(WireModel):
    id: str
    proxy: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ProxyResponse(WireModel):
    id: str
    ok: bool = True
    value: Any = None
    done: bool = False
    error: str | None = None


_REQUEST_ADAPTER = TypeAdapter(ReplRequest)


def parse_request(data: str | dict[str, Any]) -> ReplRequest:
    if isinstance(data, str):
        return _REQUEST_ADAPTER.validate_json(data)
    return _REQUEST_ADAPTER.validate_python(data)


def parse_client_message(data: str | dict[str, Any]) -> ReplResponse | ProxyCall:
    if isinstance(data, str):
        raw = TypeAdapter(dict[str, Any]).validate_json(data)
    else:
        raw = data
    if "proxy" in raw:
        return ProxyCall.model_validate(raw)
    return ReplResponse.model_validate(raw)


def dump_message(message: WireModel) -> str:
    return message.model_dump_json(exclude_none=True)


__all__ = [
    "BaseRequest",
    "CapabilitiesRequest",
    "CapabilityMap",
    "InjectImportRequest",
    "InjectProxyRequest",
    "InjectRequest",
    "InjectSourceRequest",
    "PingRequest",
    "ProxyCall",
    "ProxyResponse",
    "RemoveRequest",
    "ReplRequest",
    "ReplResponse",
    "RunRequest",
    "SetEnvRequest",
    "WireModel",
    "dump_message",
    "parse_client_message",
    "parse_request",
]
