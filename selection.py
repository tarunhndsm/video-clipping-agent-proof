"""Sentence-aligned, evidence-labelled clip selection; no model API calls."""
from collections import Counter
from dataclasses import dataclass
from bisect import bisect_right
import math
import re


class ClipError(ValueError):
    pass


@dataclass(frozen=True)
class Sentence:
    start: float
    end: float
    text: str
    words: tuple


STOP = set('a an the this that these those is are was were be been being to of in on at '
           'and or but for with as it its we you they i he she my our your their '
           'can could should would will have has had do does did not so if then '
           'from by about into just more most very also'.split())


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ClipError('Invalid ' + label)
    return float(value)


def sentences(words, duration):
    duration = number(duration, 'duration')
    if not 0 < duration <= 3600 or not isinstance(words, list) or len(words) > 100000:
        raise ClipError('Expected speech of at most 60 minutes')
    complete, group, previous_start = [], [], -1.0
    for item in words:
        if not isinstance(item, dict):
            raise ClipError('Invalid word timestamp')
        a, b = number(item.get('start'), 'word start'), number(item.get('end'), 'word end')
        text = item.get('text')
        if not isinstance(text, str) or not text.strip() or len(text) > 200:
            raise ClipError('Invalid transcript word')
        if not 0 <= a < b <= duration + 0.05 or a < previous_start:
            raise ClipError('Non-monotonic or out-of-range word timestamp')
        previous_start = a
        group.append({'start': a, 'end': min(b, duration), 'text': text.strip()})
        if re.search(r'[.!?]["\u201d\u2019\')\]]*$', text.strip()):
            complete.append(Sentence(group[0]['start'], group[-1]['end'],
                                     ' '.join(w['text'] for w in group), tuple(group)))
            group = []
    # An unfinished tail is not falsely labelled a complete sentence.
    if not complete:
        raise ClipError('No complete spoken sentences found; provide a corrected transcript')
    aligned = []
    for i, item in enumerate(complete):
        before = complete[i - 1].words[-1]['end'] if i else 0.0
        after = (complete[i + 1].words[0]['start'] if i + 1 < len(complete)
                 else group[0]['start'] if group else duration)
        if before > item.start + 0.02 or after < item.end - 0.02:
            raise ClipError('Overlapping sentence timestamps require a corrected transcript')
        aligned.append(Sentence(max(0, item.start - 0.20, (before + item.start) / 2),
                                min(duration, item.end + 0.35, (item.end + after) / 2),
                                item.text, item.words))
    return aligned


def terms(text):
    return [x for x in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if x not in STOP]


def choose(items, count=3, minimum=12.0, maximum=45.0, topics=(), heatmap=()):
    if type(count) is not int or not 1 <= count <= 10:
        raise ClipError('Clip count must be 1 to 10')
    minimum, maximum = number(minimum, 'minimum'), number(maximum, 'maximum')
    if not 3 <= minimum <= maximum <= 90:
        raise ClipError('Clip lengths must be between 3 and 90 seconds')
    if not items:
        raise ClipError('No candidate sentences')
    if not isinstance(topics, (tuple, list)) or any(not isinstance(t, str) for t in topics):
        raise ClipError('Topics must be text')
    peaks = []
    if not isinstance(heatmap, (list, tuple)) or len(heatmap) > 10000:
        raise ClipError('Invalid retention data')
    for p in heatmap:
        if not isinstance(p, dict):
            raise ClipError('Invalid retention interval')
        a, b, value = (number(p.get(k), k) for k in ('start_time', 'end_time', 'value'))
        if not 0 <= a < b or not 0 <= value <= 1:
            raise ClipError('Invalid retention interval')
        peaks.append((a, b, value))
    frequency = Counter(x for s in items for x in terms(s.text))
    wanted = set(x for topic in topics for x in terms(topic))
    candidates = []
    for i, first in enumerate(items):
        group = []
        targets = [minimum + (maximum - minimum) * f for f in (0, 1/3, 2/3, 1)]
        target_index = 0
        for j in range(i, len(items)):
            last = items[j]
            group.append(last)
            length = last.end - first.start
            if length > maximum:
                break
            if target_index >= len(targets) or length < targets[target_index]:
                continue
            while target_index < len(targets) and length >= targets[target_index]:
                target_index += 1
            text = ' '.join(s.text for s in group)
            tokens = terms(text)
            if not tokens:
                continue
            topic_hits = sum(t in wanted for t in tokens)
            distinctive = sorted(set(tokens), key=lambda t: (frequency[t], t))[:4]
            lexical = sum(1 / frequency[t] for t in set(tokens)) / max(len(tokens), 1)
            peak = max((v for a, b, v in peaks if a < last.end and b > first.start), default=0.0)
            # Retention dominates when supplied; lexical variety breaks ties.
            score = peak * 1000 + topic_hits * 10 + lexical
            mode = 'retention' if peaks else 'content_analysis_heuristic'
            reason = ('Overlaps supplied replay/retention data with peak %.3f; ' % peak if peaks else
                      'No replay/retention data was supplied. Content analysis uses transcript terms; ')
            reason += 'complete passage includes ' + ', '.join('"' + t + '"' for t in distinctive) + '.'
            if wanted:
                reason += ' Topic keyword occurrences: %d.' % topic_hits
            candidates.append({'start': first.start, 'end': last.end, 'text': text,
                               'sentence_range': (i, j + 1),
                               'score': score, 'selection_mode': mode, 'reason': reason,
                               'retention_peak': peak if peaks else None})
    ordered = sorted(candidates, key=lambda c: (c['end'], c['start']))
    ends = [c['end'] for c in ordered]
    previous = [bisect_right(ends, c['start'], 0, i) for i, c in enumerate(ordered)]
    # Exact-count weighted interval scheduling avoids a greedy highlight blocking
    # the requested number of otherwise feasible clips.
    dp = [[0.0] * (len(ordered) + 1)]
    for k in range(1, count + 1):
        row = [-math.inf] * (len(ordered) + 1)
        for n, candidate in enumerate(ordered, 1):
            row[n] = max(row[n - 1], dp[k - 1][previous[n - 1]] + candidate['score'])
        dp.append(row)
    if not math.isfinite(dp[count][-1]):
        feasible = max(k for k, row in enumerate(dp) if math.isfinite(row[-1]))
        raise ClipError('Only %d non-overlapping complete clips fit; reduce count or length' % feasible)
    chosen, k, n = [], count, len(ordered)
    while k:
        if dp[k][n] == dp[k][n - 1]:
            n -= 1
            continue
        candidate = dict(ordered[n - 1])
        a, b = candidate.pop('sentence_range')
        candidate['words'] = [w for s in items[a:b] for w in s.words]
        chosen.append(candidate)
        n = previous[n - 1]
        k -= 1
    return sorted(chosen, key=lambda c: c['start'])
