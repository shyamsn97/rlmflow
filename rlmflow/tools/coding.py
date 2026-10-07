"""Coding-agent child submission schema.

The engine renders whatever object a child ``submit()``s; this schema is the
coding CLI's default for launched children, not a library-wide submission shape.
Needle, evals, and other ``Flow``s stay untyped unless they set
``AgentConfig.child_output_schema`` themselves.
"""

CODING_HANDOFF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["done", "diff_summary", "concerns", "followups"],
    "properties": {
        "done": {
            "type": "boolean",
            "description": ("True when this slice is complete enough for the parent to integrate."),
        },
        "diff_summary": {
            "type": "string",
            "description": "What changed on disk and why, in a few sentences.",
        },
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Risks, incomplete bits, or tests that still fail.",
        },
        "followups": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Work this child did not do that a later child should.",
        },
    },
}


__all__ = ["CODING_HANDOFF_SCHEMA"]
