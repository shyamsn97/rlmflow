"""Typed JSON-line protocol for lightweight Python workers."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseRequest(WireModel):
    id: str
    cmd: str
    tenant_id: str


class PingRequest(BaseRequest):
    cmd: Literal["ping"] = "ping"


class RunRequest(BaseRequest):
    cmd: Literal["run"] = "run"
    code: str
    binding: str


class InjectRequest(BaseRequest):
    cmd: Literal["inject"] = "inject"
    name: str
    value: str


class RetrieveRequest(BaseRequest):
    cmd: Literal["retrieve"] = "retrieve"
    name: str


class RemoveRequest(BaseRequest):
    cmd: Literal["remove"] = "remove"
    name: str


class InjectProxyRequest(BaseRequest):
    cmd: Literal["inject_proxy"] = "inject_proxy"
    name: str
    is_async: bool = False


ReplRequest = Annotated[
    PingRequest | RunRequest | InjectRequest | RetrieveRequest | RemoveRequest | InjectProxyRequest,
    Field(discriminator="cmd"),
]


class ReplResponse(WireModel):
    id: str
    ok: bool = True
    output: str = ""
    errored: bool = False
    error: str | None = None
    value: JsonValue = None
    env: dict[str, JsonValue] | None = None


class ProxyCall(WireModel):
    id: str
    run_id: str
    tenant_id: str
    proxy: str
    call_id: int
    args: list[JsonValue]
    kwargs: dict[str, JsonValue]


class ProxyResponse(WireModel):
    id: str
    ok: bool = True
    value: str | None = None
    done: bool = False
    error: str | None = None


_REQUEST = TypeAdapter(ReplRequest)


def parse_host_message(data: str) -> ReplRequest | ProxyResponse:
    raw = TypeAdapter(dict[str, Any]).validate_json(data)
    if "cmd" in raw:
        return _REQUEST.validate_python(raw)
    return ProxyResponse.model_validate(raw)


def parse_client_message(data: str) -> ReplResponse | ProxyCall:
    raw = TypeAdapter(dict[str, Any]).validate_json(data)
    if "proxy" in raw:
        return ProxyCall.model_validate(raw)
    return ReplResponse.model_validate(raw)


def dump_message(message: WireModel) -> str:
    return message.model_dump_json(exclude_none=True)


__all__ = [
    "InjectProxyRequest",
    "InjectRequest",
    "PingRequest",
    "ProxyCall",
    "ProxyResponse",
    "RemoveRequest",
    "ReplRequest",
    "ReplResponse",
    "RetrieveRequest",
    "RunRequest",
    "WireModel",
    "dump_message",
    "parse_client_message",
    "parse_host_message",
]
