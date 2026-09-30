"""OpenRouter chat completions with tool calling (OpenAI-compatible API)."""
import copy
import time

from .config import OPENROUTER_BASE_URL, api_key


class LLM:
    def __init__(self, model, max_output_tokens=8000, temperature=None, reasoning_effort='medium'):
        from openai import OpenAI
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key(),
                             default_headers={'X-Title': 'PKAgent'})
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.usage = dict(calls=0, prompt_tokens=0, completion_tokens=0, cached_tokens=0, reasoning_tokens=0,
                          cost_usd=0.)

    @property
    def anthropic(self):
        return self.model.startswith('anthropic/')

    def _prepare(self, messages):
        """Prompt-cache breakpoints for Anthropic models: the system prompt, the end of the previous turn's input
        (read back from the cache) and the latest message (written to the cache for the next turn)."""
        if not self.anthropic:
            return messages
        msgs = copy.deepcopy(messages)
        last_assistant = max((i for i, m in enumerate(msgs) if m['role'] == 'assistant'), default=None)
        inputs = [i for i, m in enumerate(msgs) if m['role'] in ('user', 'tool')]
        marks = {0}
        if inputs:
            marks.add(inputs[-1])
        if last_assistant is not None:
            before = [i for i in inputs if i < last_assistant]
            if before:
                marks.add(before[-1])
        for i in sorted(marks):
            m = msgs[i]
            content = m.get('content')
            if isinstance(content, str) and content:
                m['content'] = [dict(type='text', text=content, cache_control=dict(type='ephemeral'))]
            elif isinstance(content, list) and content:
                for part in content[::-1]:
                    if part.get('type') == 'text':
                        part['cache_control'] = dict(type='ephemeral')
                        break
        return msgs

    def chat(self, messages, tools):
        body = dict(model=self.model, messages=self._prepare(messages), tools=tools, tool_choice='auto',
                    max_tokens=self.max_output_tokens)
        if self.temperature is not None:
            body['temperature'] = self.temperature
        extra = dict(usage=dict(include=True))
        if self.reasoning_effort:
            extra['reasoning'] = dict(effort=self.reasoning_effort)
        delay = 5.
        last = None
        for attempt in range(8):
            try:
                resp = self.client.chat.completions.create(**body, extra_body=extra, timeout=900)
                if not resp.choices:
                    raise RuntimeError(f'empty response: {getattr(resp, "error", None)}')
                break
            except Exception as e:                               # noqa: BLE001 - retried, then raised
                last = e
                status = getattr(e, 'status_code', None)
                if status in (400, 401, 402, 403, 404) and attempt >= 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120.)
        else:
            raise RuntimeError(f'LLM request failed after retries: {last}')
        self._account(resp)
        msg = resp.choices[0].message
        out = dict(role='assistant', content=msg.content or '')
        if msg.tool_calls:
            out['tool_calls'] = [dict(id=c.id, type='function',
                                      function=dict(name=c.function.name, arguments=c.function.arguments or '{}'))
                                 for c in msg.tool_calls]
        extra_fields = getattr(msg, 'model_extra', None) or {}
        if extra_fields.get('reasoning_details'):
            out['reasoning_details'] = extra_fields['reasoning_details']
        reasoning = extra_fields.get('reasoning')
        return out, dict(finish_reason=resp.choices[0].finish_reason, reasoning=reasoning)

    def _account(self, resp):
        u = getattr(resp, 'usage', None)
        self.usage['calls'] += 1
        if u is None:
            return
        self.usage['prompt_tokens'] += int(getattr(u, 'prompt_tokens', 0) or 0)
        self.usage['completion_tokens'] += int(getattr(u, 'completion_tokens', 0) or 0)
        details = getattr(u, 'prompt_tokens_details', None)
        if details is not None:
            self.usage['cached_tokens'] += int(getattr(details, 'cached_tokens', 0) or 0)
        cdetails = getattr(u, 'completion_tokens_details', None)
        if cdetails is not None:
            self.usage['reasoning_tokens'] += int(getattr(cdetails, 'reasoning_tokens', 0) or 0)
        cost = (getattr(u, 'model_extra', None) or {}).get('cost')
        if cost is None:
            cost = getattr(u, 'cost', None)
        if cost is not None:
            self.usage['cost_usd'] += float(cost)
