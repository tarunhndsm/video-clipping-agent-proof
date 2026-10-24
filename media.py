"""Local, bounded media operations. Source files are never modified."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import textwrap

from selection import ClipError, number

FORMATS = {'vertical': (720, 1280), 'square': (720, 720), 'landscape': (1280, 720)}
SPEECH_MODEL = 'small.en'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command(args, timeout=600, cwd=None):
    try:
        p = subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise ClipError('Media operation unavailable or timed out') from None
    if p.returncode:
        # Do not leak source URLs, private paths or dependency diagnostics.
        raise ClipError('Media operation failed; check supported source and formats')
    return p.stdout


def probe(source):
    path = Path(source).resolve()
    if not path.is_file() or path.stat().st_size > 1024 ** 3:
        raise ClipError('Source must be a local file of at most 1 GiB')
    raw = command(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                   '-show_streams', '-show_format', '-of', 'json', str(path)], timeout=30)
    try:
        data = json.loads(raw)
        duration = number(float(data['format']['duration']), 'media duration')
    except (KeyError, TypeError, ValueError):
        raise ClipError('Source duration could not be established') from None
    streams = data.get('streams', [])
    videos = [s for s in streams if s.get('codec_type') == 'video']
    if not videos or not 0 < duration <= 3600:
        raise ClipError('Expected a video lasting at most 60 minutes')
    video = videos[0]
    if video.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ClipError('HDR footage needs an explicit color conversion before this proof')
    return {'duration': duration, 'width': video['width'], 'height': video['height'],
            'audio': any(s.get('codec_type') == 'audio' for s in streams),
            'video_codec': video.get('codec_name'),
            'audio_codecs': [s.get('codec_name') for s in streams if s.get('codec_type') == 'audio']}


def transcribe(source, cache):
    metadata = probe(source)
    if not metadata['audio']:
        raise ClipError('This speech-focused proof requires an audio track')
    key = digest(source)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = cache / (key + '-' + SPEECH_MODEL + '-v2.json')
    if target.exists():
        return json.loads(target.read_text())
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from faster_whisper import WhisperModel
    import numpy as np
    model = WhisperModel(SPEECH_MODEL, device='cpu', compute_type='int8', cpu_threads=4,
                         num_workers=1)
    # Decode through the same bounded FFmpeg path as rendering. This also avoids
    # version-dependent PyAV metadata options in speech-library file decoding.
    pcm = command(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-protocol_whitelist', 'file,pipe', '-i', str(Path(source).resolve()),
                   '-map', '0:a:0', '-vn', '-t', str(metadata['duration']),
                   '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'])
    audio = np.frombuffer(pcm, dtype='<f4')
    result, info = model.transcribe(audio, language='en',
                                    beam_size=5, word_timestamps=True, vad_filter=False,
                                    condition_on_previous_text=True)
    words = []
    for segment in result:
        if segment.no_speech_prob > 0.8:
            continue
        for word in segment.words or []:
            a, b = max(0.0, word.start), min(metadata['duration'], word.end)
            if a < b and word.word.strip():
                words.append({'text': word.word.strip(), 'start': round(a, 4), 'end': round(b, 4)})
    if not words:
        raise ClipError('No usable English speech found; supply a corrected transcript')
    value = {'source_sha256': key, 'speech_model': 'faster-whisper/' + SPEECH_MODEL,
             'language': info.language, 'duration': metadata['duration'],
             'vad_filter': False, 'condition_on_previous_text': True, 'words': words}
    temporary = target.with_suffix('.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    return value


def caption_cues(words, start, end):
    cues, group = [], []
    for w in words:
        if w['start'] < start - 0.01 or w['end'] > end + 0.01:
            raise ClipError('A clip boundary cuts through a transcript word')
        group.append(w)
        if len(group) >= 5 or w['end'] - group[0]['start'] >= 2.2 or w['text'].endswith(('.', '?', '!')):
            cues.append({'start': max(0, group[0]['start'] - start),
                         'end': min(end - start, group[-1]['end'] - start + 0.08),
                         'text': ' '.join(x['text'] for x in group)})
            group = []
    if group:
        cues.append({'start': max(0, group[0]['start'] - start),
                     'end': min(end - start, group[-1]['end'] - start + 0.08),
                     'text': ' '.join(x['text'] for x in group)})
    for i, cue in enumerate(cues[:-1]):
        cue['end'] = min(cue['end'], cues[i + 1]['start'])
    return [c for c in cues if c['end'] > c['start']]


def srt(cues):
    def stamp(t):
        millis = max(0, round(t * 1000))
        hours, millis = divmod(millis, 3600000)
        minutes, millis = divmod(millis, 60000)
        seconds, millis = divmod(millis, 1000)
        return '%02d:%02d:%02d,%03d' % (hours, minutes, seconds, millis)
    return '\n\n'.join('%d\n%s --> %s\n%s' % (i, stamp(c['start']), stamp(c['end']), c['text'])
                       for i, c in enumerate(cues, 1)) + '\n'


def caption_stream(cues, duration, width, folder):
    from PIL import Image, ImageDraw, ImageFont
    height = 270
    font_size = 44 if width <= 720 else 52
    font = ImageFont.load_default(size=font_size)
    blank = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    blank.save(folder / 'blank.png')
    timeline, cursor = [], 0.0
    for i, cue in enumerate(cues):
        if cue['start'] > cursor:
            timeline.append(('blank.png', cue['start'] - cursor))
        image = blank.copy()
        draw = ImageDraw.Draw(image)
        wrapped, line = [], ''
        for word in cue['text'].split():
            candidate = (line + ' ' + word).strip()
            if draw.textlength(candidate, font=font) > width - 80 and line:
                wrapped.append(line)
                line = word
            else:
                line = candidate
        if line:
            wrapped.append(line)
        if len(wrapped) > 3 or any(draw.textlength(t, font=font) > width - 60 for t in wrapped):
            raise ClipError('Caption text does not fit; shorten the corrected transcript cues')
        text = '\n'.join(wrapped)
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=10, align='center')
        text_width, text_height = box[2] - box[0], box[3] - box[1]
        x, y = (width - text_width) / 2, 52
        draw.rounded_rectangle((x - 20, y - 16, x + text_width + 20, y + text_height + 26),
                               radius=16, fill=(8, 14, 24, 226))
        draw.multiline_text((x, y - box[1]), text, font=font, fill='white',
                            spacing=10, align='center', stroke_width=1, stroke_fill='black')
        filename = 'cue-%05d.png' % i
        image.save(folder / filename)
        timeline.append((filename, cue['end'] - cue['start']))
        cursor = cue['end']
    if cursor < duration:
        timeline.append(('blank.png', duration - cursor))
    if not timeline:
        timeline = [('blank.png', duration)]
    rows = ['ffconcat version 1.0']
    for filename, seconds in timeline:
        if seconds > 0:
            rows.extend(["file '%s'" % filename, 'duration %.6f' % seconds])
    rows.append("file '%s'" % timeline[-1][0])
    path = folder / 'captions.ffconcat'
    path.write_text('\n'.join(rows) + '\n')
    return path


def render(source, clip, fmt, output, captions=True, layout='fit'):
    if fmt not in FORMATS or layout not in ('fit', 'center_crop'):
        raise ClipError('Unsupported output format or layout')
    output = Path(output).resolve()
    if output.exists():
        raise ClipError('Output already exists; use a new revision directory')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    width, height = FORMATS[fmt]
    duration = clip['end'] - clip['start']
    cues = caption_cues(clip['words'], clip['start'], clip['end']) if captions else []
    with tempfile.TemporaryDirectory(prefix='.render-', dir=output.parent) as tmp:
        folder = Path(tmp)
        args = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                '-protocol_whitelist', 'file,pipe', '-ss', '%.6f' % clip['start'],
                '-i', str(Path(source).resolve())]
        if captions:
            caption_path = caption_stream(cues, duration, width, folder)
            args += ['-f', 'concat', '-safe', '1', '-i', str(caption_path)]
        if layout == 'fit':
            scale = ('scale=%d:%d:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                     'pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=0x101723,setsar=1' %
                     (width, height, width, height))
        else:
            scale = ('scale=%d:%d:force_original_aspect_ratio=increase:force_divisible_by=2,'
                     'crop=%d:%d,setsar=1' % (width, height, width, height))
        filters = '[0:v]%s[base];[base][1:v]overlay=0:H-h-80:format=auto[v]' % scale if captions else '[0:v]%s[v]' % scale
        temporary_output = folder / 'encoded.mp4'
        args += ['-filter_complex_threads', '1', '-filter_complex', filters,
                 '-map', '[v]', '-map', '0:a:0?', '-t', '%.6f' % duration,
                 '-r', '30', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '21',
                 '-pix_fmt', 'yuv420p', '-threads', '2', '-c:a', 'aac', '-b:a', '128k',
                 '-ar', '48000', '-movflags', '+faststart', '-map_metadata', '-1', str(temporary_output)]
        command(args, timeout=max(120, int(duration * 12)))
        measured = probe(temporary_output)
        if (measured['width'], measured['height']) != (width, height) or abs(measured['duration'] - duration) > 0.15:
            raise ClipError('Rendered output failed dimension or duration verification')
        os.link(temporary_output, output)  # Atomic no-clobber publication.
    return {'file': output.name, 'sha256': digest(output), 'bytes': output.stat().st_size,
            'format': fmt, 'width': width, 'height': height, 'duration': measured['duration'],
            'captions_burned_in': captions, 'layout': layout, 'caption_cues': cues}
