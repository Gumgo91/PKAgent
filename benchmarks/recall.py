"""Statements of a run's language model that refer to prior knowledge of a published dataset or its analysis.

One definition used throughout the evaluation: a word-bounded search (so 'classical', as in classical
allometric exponents, does not match 'classic') of the assistant text, the tool-call arguments and the reasoning
summaries of every turn of transcript.jsonl.
"""
import json
import re

RECALL = re.compile(r'\bclassic\b|well[- ]known|textbook|nonmem example|published|literature|grasela|donn\b|minto',
                    re.IGNORECASE)


def turn_text(turn):
    parts = [turn.get('content') or '', json.dumps(turn.get('tool_calls') or '', ensure_ascii=False)]
    reasoning = turn.get('reasoning')
    if reasoning:
        parts.append(reasoning if isinstance(reasoning, str) else json.dumps(reasoning, ensure_ascii=False))
    return ' '.join(parts)


def recalled(run_dir, context=120):
    """Matches in a run: turn, matched term, and the surrounding text."""
    out = []
    path = run_dir / 'transcript.jsonl'
    if not path.exists():
        return out
    for line in path.read_text(encoding='utf-8').splitlines():
        turn = json.loads(line)
        text = turn_text(turn)
        for m in RECALL.finditer(text):
            out.append(dict(turn=turn['turn'], term=m.group(0),
                            context=text[max(0, m.start() - context):m.end() + context].replace('\\n', ' ')))
    return out
