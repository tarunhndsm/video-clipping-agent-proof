"""File-based clipping agent, with explicit rights and an immutable revision log."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone

from media import FORMATS, digest, probe, render, srt, transcribe
from selection import ClipError, choose, sentences

VERSION = '0.1.0'


def emit(value, as_json=False):
    if as_json:
        print(json.dumps(value, ensure_ascii=False))
    else:
        # This boundary emits only scalar values, a subset of the TOON format.
        for key, item in value.items():
            print(key + ': ' + json.dumps(item, ensure_ascii=False))


class Parser(argparse.ArgumentParser):
    def error(self, message):
        emit({'ok': False, 'code': 'USAGE', 'error': message,
              'help': self.format_help().strip()}, '--json' in sys.argv)
        raise SystemExit(2)


def verified_assets(folder, manifest):
    for clip in manifest['clips']:
        for asset in clip['assets']:
            if (not re.fullmatch(r'clip-\d{2}-(vertical|square|landscape)\.mp4', asset['file'])
                    or digest(folder / asset['file']) != asset['sha256']):
                raise ClipError('Existing delivery was modified; reconcile before continuing')
            if asset.get('caption_file') and digest(folder / asset['caption_file']) != asset['caption_sha256']:
                raise ClipError('Existing captions were modified; reconcile before continuing')
    for key, name in [('transcript_sha256', 'transcript.json'), ('report_sha256', 'REPORT.md')]:
        if manifest.get(key) and digest(folder / name) != manifest[key]:
            raise ClipError('Existing report or transcript was modified; reconcile before continuing')
    return manifest


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 20_000_000:
        raise ClipError('JSON input exceeds 20 MB')
    return json.loads(path.read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def brief(value):
    if not isinstance(value, dict) or value.get('rights_confirmed') is not True:
        raise ClipError('Confirm source rights in the brief before processing')
    if type(value.get('clip_count')) is not int or not 1 <= value['clip_count'] <= 10:
        raise ClipError('Specify 1 to 10 clips')
    formats = value.get('formats')
    if not isinstance(formats, list) or not formats or any(f not in FORMATS for f in formats):
        raise ClipError('Supported formats: vertical, square, landscape; none may be omitted silently')
    if len(formats) != len(set(formats)):
        raise ClipError('Duplicate format requested')
    if type(value.get('captions')) is not bool:
        raise ClipError('Choose captions true or false')
    if value.get('layout', 'fit') not in ('fit', 'center_crop'):
        raise ClipError('Choose fit or center_crop layout')
    return value


def report(manifest):
    lines = ['# Video clipping delivery', '',
             'Source SHA256: ' + manifest['source_sha256'],
             'Selection: ' + manifest['selection_mode'],
             'Retention provenance: ' + manifest['retention_provenance'], '',
             'This tool uses transcript terms and any supplied retention data. '
             'It does not predict virality. Speech recognition may need correction; '
             'review the included transcript and play the clips before publishing.', '',
             'Fit layout preserves the full frame with padding. Center crop is opt-in '
             'and does not track speakers or faces.', '']
    for clip in manifest['clips']:
        lines += ['## ' + clip['id'],
                  'Source %.3f–%.3f seconds.' % (clip['start'], clip['end']),
                  clip['reason'], '', clip['text'], '']
        for asset in clip['assets']:
            lines.append('- %s: %s (%d×%d), SHA256 %s' %
                         (asset['format'], asset['file'], asset['width'],
                          asset['height'], asset['sha256']))
        lines.append('')
    return '\n'.join(lines)


def create(source, request, output, transcript=None, retention=None, cache='.cache'):
    request = brief(request)
    source, output = Path(source).resolve(), Path(output).resolve()
    intent = {'brief': request, 'transcript': digest(transcript) if transcript else 'auto-small-en-v2',
              'retention': digest(retention) if retention else None}
    signature = hashlib.sha256(json.dumps(intent, sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not (output / 'manifest.json').is_file():
            raise ClipError('Delivery directory exists; use a new output directory')
        saved = read_json(output / 'manifest.json')
        if saved.get('input_signature') == signature and saved.get('source_sha256') == digest(source):
            return {**verified_assets(output, saved), '_no_op': True}
        raise ClipError('Delivery directory exists with different inputs; use revise or a new directory')
    metadata = probe(source)
    source_hash = digest(source)
    speech = read_json(transcript) if transcript else transcribe(source, cache)
    if speech.get('source_sha256') != source_hash:
        raise ClipError('Corrected transcript must match the source SHA256')
    timing = sentences(speech.get('words'), metadata['duration'])
    heatmap, provenance = [], 'None supplied; transcript content analysis used.'
    if retention:
        supplied = read_json(retention)
        if supplied.get('source_sha256') != source_hash:
            raise ClipError('Retention metadata must match this source SHA256')
        provenance = supplied.get('provenance')
        if not isinstance(provenance, str) or not provenance.strip() or len(provenance) > 500:
            raise ClipError('State the provenance of retention metadata')
        heatmap = supplied.get('heatmap')
        if not heatmap:
            raise ClipError('Retention file has no usable intervals')
        if any(not isinstance(p, dict) or p.get('end_time', 1e10) > metadata['duration']
               for p in heatmap):
            raise ClipError('Retention interval exceeds the source')
    selected = choose(timing, request['clip_count'], request.get('minimum_seconds', 12),
                      request.get('maximum_seconds', 45), request.get('topics', []), heatmap)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.delivery-', dir=output.parent) as tmp:
        staging = Path(tmp)
        manifest = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
                    'input_signature': signature,
                    'source_sha256': source_hash, 'source': metadata, 'brief': request,
                    'selection_mode': selected[0]['selection_mode'],
                    'retention_provenance': provenance, 'clips': []}
        save_json(staging / 'transcript.json', speech)
        manifest['transcript_sha256'] = digest(staging / 'transcript.json')
        for i, selected_clip in enumerate(selected, 1):
            clip = {**selected_clip, 'id': 'clip-%02d' % i, 'assets': []}
            for fmt in request['formats']:
                path = staging / (clip['id'] + '-' + fmt + '.mp4')
                asset = render(source, clip, fmt, path, request['captions'], request.get('layout', 'fit'))
                clip['assets'].append(asset)
                if request['captions']:
                    path.with_suffix('.srt').write_text(srt(asset['caption_cues']))
                    asset.update(caption_file=path.with_suffix('.srt').name,
                                 caption_sha256=digest(path.with_suffix('.srt')))
            manifest['clips'].append(clip)
        (staging / 'REPORT.md').write_text(report(manifest) + '\n')
        manifest['report_sha256'] = digest(staging / 'REPORT.md')
        save_json(staging / 'manifest.json', manifest)
        if digest(source) != source_hash:
            raise ClipError('Source changed during processing; no delivery was published')
        os.rename(staging, output)
    return manifest


def revise(source, original, changes, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    manifest = read_json(original / 'manifest.json')
    verified_assets(original, manifest)
    if digest(source) != manifest['source_sha256']:
        raise ClipError('Use the unchanged source and a new revision directory')
    if output.exists():
        saved = read_json(output / 'manifest.json')
        prior = saved.get('revision', {})
        if prior.get('changes') == changes and prior.get('original_manifest_sha256') == digest(original / 'manifest.json'):
            return {**verified_assets(output, saved), '_no_op': True}
        raise ClipError('Revision directory exists with different inputs; use a new directory')
    if not isinstance(changes, dict) or not changes or set(changes) - {'clips'}:
        raise ClipError('Revision requires a clips map')
    edits = changes.get('clips')
    ids = {c['id'] for c in manifest['clips']}
    if not isinstance(edits, dict) or not edits or set(edits) - ids:
        raise ClipError('Revision names an unknown clip')
    speech = read_json(original / 'transcript.json')
    timing = sentences(speech['words'], manifest['source']['duration'])
    for clip in manifest['clips']:
        for asset in clip['assets']:
            name = asset['file']
            if not re.fullmatch(r'clip-\d{2}-(vertical|square|landscape)\.mp4', name):
                raise ClipError('Unexpected delivery filename')
            if digest(original / name) != asset['sha256']:
                raise ClipError('Original delivery was modified; reconcile before revision')
    original_hashes = {p.name: digest(p) for p in original.iterdir() if p.is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.revision-', dir=output.parent) as tmp:
        staging = Path(tmp)
        for path in original.iterdir():
            if not path.is_file() or path.is_symlink():
                raise ClipError('Unexpected entry in original delivery')
            shutil.copy2(path, staging / path.name)
        for clip in manifest['clips']:
            if clip['id'] not in edits:
                continue
            change = edits[clip['id']]
            if not isinstance(change, dict) or not change or set(change) - {'layout', 'captions', 'sentences'}:
                raise ClipError('Supported revision fields: layout, captions, sentences')
            layout = change.get('layout', clip['assets'][0]['layout'])
            captions = change.get('captions', clip['assets'][0]['captions_burned_in'])
            if layout not in ('fit', 'center_crop') or type(captions) is not bool:
                raise ClipError('Invalid revised layout or caption choice')
            if 'sentences' in change:
                span = change['sentences']
                if (not isinstance(span, list) or len(span) != 2 or
                    any(type(x) is not int for x in span) or not 0 <= span[0] < span[1] <= len(timing)):
                    raise ClipError('Sentence span is [first, exclusive_last], using zero-based indices')
                group = timing[span[0]:span[1]]
                if not 3 <= group[-1].end - group[0].start <= 90:
                    raise ClipError('Revised clip must last 3 to 90 seconds')
                clip.update(start=group[0].start, end=group[-1].end,
                            text=' '.join(s.text for s in group),
                            words=[w for s in group for w in s.words],
                            reason='Buyer-requested complete sentence span %d:%d.' % tuple(span),
                            selection_mode='buyer_revision', retention_peak=None)
            replacement = []
            for asset in clip['assets']:
                path = staging / asset['file']
                path.unlink()
                caption_path = path.with_suffix('.srt')
                caption_path.unlink(missing_ok=True)
                updated = render(source, clip, asset['format'], path, captions, layout)
                replacement.append(updated)
                if captions:
                    caption_path.write_text(srt(updated['caption_cues']))
                    updated.update(caption_file=caption_path.name, caption_sha256=digest(caption_path))
            clip['assets'] = replacement
        unchanged = {}
        for clip in manifest['clips']:
            if clip['id'] in edits:
                continue
            for asset in clip['assets']:
                for name in [asset['file'], str(Path(asset['file']).with_suffix('.srt'))]:
                    if name in original_hashes:
                        if digest(staging / name) != original_hashes[name]:
                            raise ClipError('Unrequested clip changed')
                        unchanged[name] = original_hashes[name]
        manifest['revision'] = {'changed_clip_ids': sorted(edits), 'unchanged_files': unchanged,
                                'original_manifest_sha256': digest(original / 'manifest.json'),
                                'changes': changes}
        manifest.pop('input_signature', None)
        (staging / 'REPORT.md').write_text(report(manifest) + '\n')
        manifest['report_sha256'] = digest(staging / 'REPORT.md')
        save_json(staging / 'manifest.json', manifest)
        if digest(source) != manifest['source_sha256']:
            raise ClipError('Source changed during revision; no delivery was published')
        os.rename(staging, output)
    return manifest


def main():
    if not sys.argv[1:]:
        emit({'bin': str(Path(__file__).resolve()).replace(str(Path.home()), '~', 1),
              'description': 'Create and revise sentence-aligned clips locally.',
              'deliveries_in_current_directory': sum(1 for _ in Path.cwd().glob('*/manifest.json')),
              'help': 'python agent.py create --source <video> --brief <brief.json> --output <new-directory>'})
        return
    parser = Parser(description='Create and revise sentence-aligned clips locally.')
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('create')
    make.add_argument('--source', required=True)
    make.add_argument('--brief', required=True)
    make.add_argument('--output', required=True)
    make.add_argument('--transcript')
    make.add_argument('--retention')
    make.add_argument('--cache', default='.cache')
    edit = sub.add_parser('revise')
    edit.add_argument('--source', required=True)
    edit.add_argument('--original', required=True)
    edit.add_argument('--changes', required=True)
    edit.add_argument('--output', required=True)
    ingest = sub.add_parser('ingest')
    ingest.add_argument('--url', required=True)
    ingest.add_argument('--output', required=True)
    ingest.add_argument('--rights-confirmed', action='store_true')
    for command_parser in (make, edit, ingest):
        command_parser.add_argument('--json', action='store_true', help='Emit JSON instead of TOON')
    make.epilog = ('Example: python agent.py create --source input.mp4 --brief brief.json --output delivery\n'
                   'Example: python agent.py create --source input.mp4 --brief brief.json --transcript corrected.json --output delivery-2')
    edit.epilog = ('Example: python agent.py revise --source input.mp4 --original delivery --changes changes.json --output revision\n'
                   'Example: python agent.py revise --source input.mp4 --original delivery --changes captions.json --output revision-2 --json')
    ingest.epilog = ('Example: python agent.py ingest --url https://youtu.be/<video-id> --output input --rights-confirmed\n'
                     'Example: python agent.py ingest --url https://www.youtube.com/watch?v=<video-id> --output input-2 --rights-confirmed --json')
    args = parser.parse_args()
    try:
        if args.command == 'ingest':
            from ingest import download
            emit({'ok': True, **download(args.url, args.output, args.rights_confirmed)}, args.json)
            return
        if args.command == 'create':
            value = create(args.source, read_json(args.brief), args.output,
                           args.transcript, args.retention, args.cache)
        else:
            value = revise(args.source, args.original, read_json(args.changes), args.output)
        emit({'ok': True, 'clips': len(value['clips']),
              'files': sum(len(c['assets']) for c in value['clips']),
              'no_op': bool(value.get('_no_op')),
              'help': 'Review ' + str(Path(args.output) / 'REPORT.md') + ' and play the clips.'}, args.json)
    except Exception as error:
        emit({'ok': False, 'code': 'INPUT_OR_MEDIA_ERROR',
              'error': str(error) if isinstance(error, ClipError)
              else 'Input or media operation failed; no complete delivery published',
              'help': 'Correct the brief or inputs and use a new output directory.'}, args.json)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
