"""The analysis loop: the LLM calls tools until it finalizes a model."""
import base64
import json
import time
from pathlib import Path

from .config import Settings, load_env
from .llm import LLM
from .prompts import SYSTEM, task
from .report import write_report
from .session import Session
from .tools import TOOLS, dumps, execute


def _image_part(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return dict(type='image_url', image_url=dict(url=f'data:image/png;base64,{data}'))


def run(data_path, out_dir, description, knowledge=None, settings=None, objective=None, progress=print):
    """Run one analysis; returns the path of the report."""
    load_env()
    settings = settings or Settings()
    out = Path(out_dir)
    session = Session(data_path, out, settings, description)
    llm = LLM(settings.model, settings.max_output_tokens, settings.temperature, settings.reasoning_effort)
    (out / 'run.json').write_text(json.dumps(dict(
        model=settings.model, data=str(data_path), description=description, knowledge=knowledge, objective=objective,
        seed=settings.seed, budget=vars(settings.budget), reasoning_effort=settings.reasoning_effort,
        started=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1), encoding='utf-8')
    messages = [dict(role='system', content=SYSTEM),
                dict(role='user', content=task(description, knowledge, objective, settings.budget))]
    nudges, final_requested = 0, False
    try:
        while session.final is None:
            b = settings.budget
            over = (session.turns >= b.max_turns or (time.time() - session.t0) / 3600 > b.max_hours
                    or llm.usage['cost_usd'] > b.max_cost_usd or session.fits_used >= b.max_fits)
            if over:
                if final_requested:
                    progress('budget exhausted without finalize_model; stopping')
                    break
                final_requested = True
                messages.append(dict(role='user', content='The budget is exhausted. Call finalize_model now with the best '
                                                          'converged model and the report.'))
            msg, meta = llm.chat(messages, TOOLS)
            session.turns += 1
            session.llm_cost = llm.usage['cost_usd']
            messages.append(msg)
            _transcript(out, session.turns, msg, meta, llm.usage)
            if msg.get('content'):
                progress(f"[{session.turns}] {msg['content'][:400]}")
            calls = msg.get('tool_calls') or []
            if not calls:
                nudges += 1
                if nudges > 3:
                    progress('the model stopped calling tools; stopping')
                    break
                messages.append(dict(role='user', content='Continue the analysis with the tools, or call finalize_model '
                                                          'with the final model and the report.'))
                continue
            images = []
            for c in calls:
                name = c['function']['name']
                try:
                    args = json.loads(c['function']['arguments'] or '{}')
                    result, imgs = execute(session, name, args)
                except json.JSONDecodeError as e:
                    args, result, imgs = None, dict(error=f'arguments are not valid JSON: {e}'), []
                progress(f"    {name}({_brief(args)}) -> {_brief(result, 300)}")
                session.log(dict(turn=session.turns, tool=name, args=args, result=result,
                                 seconds=round(time.time() - session.t0, 1)))
                messages.append(dict(role='tool', tool_call_id=c['id'], content=dumps(result)))
                images += imgs
            if images and settings.images:
                _drop_old_images(messages, keep=KEEP_IMAGE_MESSAGES - 1)
                content = [_image_part(p) for p in images[:6]]      # text last: it carries the cache breakpoint
                content.append(dict(type='text', text='These are the plots requested above: '
                                                      + ', '.join(p.name for p in images[:6])))
                messages.append(dict(role='user', content=content))
    finally:
        (out / 'messages.json').write_text(json.dumps(_strip_images(messages), indent=1, default=str), encoding='utf-8')
        try:
            report = write_report(session, llm, settings, knowledge)
        finally:
            session.close()
    return report


KEEP_IMAGE_MESSAGES = 2      # plots stay in the conversation for this many image messages, then become text


def _drop_old_images(messages, keep):
    """Replace the images of all but the last `keep` image messages by a note (keeps the context bounded)."""
    idx = [i for i, m in enumerate(messages) if isinstance(m.get('content'), list)
           and any(p.get('type') == 'image_url' for p in m['content'])]
    for i in idx[:max(0, len(idx) - keep)]:
        text = ' '.join(p['text'] for p in messages[i]['content'] if p.get('type') == 'text')
        messages[i] = dict(role='user', content=text + ' (images removed from the conversation to save context; '
                                                       'call view_plots again if needed)')


def _brief(obj, n=160):
    s = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= n else s[:n] + '...'


def _strip_images(messages):
    out = []
    for m in messages:
        if isinstance(m.get('content'), list):
            m = dict(m, content=[p if p.get('type') != 'image_url' else dict(type='image_url', image_url='<png>')
                                 for p in m['content']])
        out.append(m)
    return out


def _transcript(out, turn, msg, meta, usage):
    with open(out / 'transcript.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(turn=turn, content=msg.get('content'), tool_calls=msg.get('tool_calls'),
                                finish_reason=meta.get('finish_reason'), reasoning=meta.get('reasoning'),
                                usage=dict(usage)), default=str) + '\n')
