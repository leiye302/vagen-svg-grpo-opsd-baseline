"""Native SVG response boundary, without hiding the registered route token."""
from dataclasses import dataclass
import re
from .safety import svg_safety

MARKER = "<RETHINK>"
_RESPONSE = re.compile(
    r"\A\s*<think>(?P<reasoning>.*?)</think>\s*"
    r"(?P<marker><RETHINK>)?\s*<answer>(?P<svg>.*?)</answer>\s*\Z", re.S)


@dataclass(frozen=True)
class Route:
    protocol_valid: bool
    triggered: bool
    marker_count: int
    reasoning: str
    svg: str
    reason: str | None


def parse_route(text):
    count = text.count(MARKER)
    match = _RESPONSE.fullmatch(text)
    if not match or any(text.count(tag) != 1 for tag in (
            "<think>", "</think>", "<answer>", "</answer>")):
        return Route(False, False, count, "", "", "invalid_protocol")
    reasoning, svg = match["reasoning"].strip(), match["svg"].strip()
    triggered = match["marker"] is not None
    if count != int(triggered) or not reasoning or not svg:
        return Route(False, False, count, reasoning, svg, "invalid_route_boundary")
    if "<svg" not in svg or "</svg>" not in svg:
        return Route(False, triggered, count, reasoning, svg, "incomplete_svg")
    safe, safety_reason = svg_safety(svg)
    if not safe:
        # Parse the entire answer, never execute only a valid-looking substring.
        return Route(False, triggered, count, reasoning, svg, safety_reason)
    return Route(True, triggered, count, reasoning, svg, None)


def decode_policy_responses(tokenizer, responses):
    """Strip transport EOS/padding IDs only; retain RETHINK and other content.

    Replaces VAGEN's skip_special_tokens=True for both arms. It is not a
    sampler, route override, forced prefix, or insertion of a missing marker.
    """
    terminal_ids = {x for x in (tokenizer.eos_token_id, tokenizer.pad_token_id)
                    if x is not None}
    result = []
    for response in responses:
        ids = response.tolist() if hasattr(response, "tolist") else list(response)
        while ids and ids[-1] in terminal_ids:
            ids.pop()
        result.append(tokenizer.decode(ids, skip_special_tokens=False))
    return result


def native_environment_text(text):
    """Make a valid optional route marker transparent to native SVG parser.

    Retain the exact original response separately in trajectory information.
    Invalid/multiple/misplaced markers are not repaired or moved. Empty text
    intentionally enters the native invalid-action path (zero format reward),
    so the native SVG fallback cannot execute a hidden valid SVG inside an
    invalid response. This strict boundary applies equally to both arms.
    """
    route = parse_route(text)
    if not route.protocol_valid:
        return ""
    return text.replace(MARKER, "", 1) if route.protocol_valid and route.triggered else text

