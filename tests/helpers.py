"""Shared test doubles for the running-only core."""

from __future__ import annotations

from rlmflow import LLMUsage


class StubLLM:
    def __init__(self, fn):
        self.fn = fn

    def chat(self, messages, **_kwargs):
        return self.fn(messages)


class UsageLLM:
    def __init__(self, reply, input_tokens=0, output_tokens=0):
        self.reply = reply
        self.last_usage = None
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def chat(self, messages, **_kwargs):
        self.last_usage = LLMUsage(self.input_tokens, self.output_tokens)
        return self.reply(messages) if callable(self.reply) else self.reply


def first_user(messages):
    return next(message["content"] for message in messages if message["role"] == "user")


def counting_replies(*replies):
    state = {"index": 0}

    def reply(_messages):
        index = min(state["index"], len(replies) - 1)
        state["index"] += 1
        return replies[index]

    return reply
