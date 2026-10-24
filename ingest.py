"""Optional public YouTube input. No account, cookies or access-control workarounds."""
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import re

from media import digest, probe
from selection import ClipError


def youtube_url(value):
    try:
        url = urlparse(value)
        if url.scheme != 'https' or url.username or url.password or url.port not in (None, 443):
            raise ValueError()
        host = (url.hostname or '').lower()
        if host == 'youtu.be':
            identifier = url.path.strip('/')
        elif host in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
            if url.path == '/watch':
                identifiers = parse_qs(url.query).get('v', [])
                identifier = identifiers[0] if len(identifiers) == 1 else ''
            elif url.path.startswith(('/shorts/', '/live/')):
                identifier = url.path.split('/')[2]
            else:
                identifier = ''
        else:
            identifier = ''
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', identifier):
            raise ValueError()
    except (ValueError, TypeError, IndexError):
        raise ClipError('Use a public HTTPS YouTube video link, or supply a local upload') from None
    return 'https://www.youtube.com/watch?v=' + identifier


class Quiet:
    def debug(self, *_):
        pass
    def warning(self, *_):
        pass
    def error(self, *_):
        pass


def download(url, destination, rights_confirmed=False):
    if rights_confirmed is not True:
        raise ClipError('Source-rights confirmation is required')
    canonical = youtube_url(url)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ClipError('Input directory exists; preserving the earlier source')
    destination.mkdir(parents=True, mode=0o700)
    from yt_dlp import YoutubeDL
    def limit(update):
        if update.get('downloaded_bytes', 0) > 1024 ** 3:
            raise ClipError('Download exceeds 1 GiB')
    options = {'format': 'bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]',
               'outtmpl': str(destination / 'source.%(ext)s'), 'merge_output_format': 'mp4',
               'noplaylist': True, 'max_downloads': 1, 'max_filesize': 1024 ** 3,
               'socket_timeout': 20, 'retries': 1, 'fragment_retries': 1,
               'concurrent_fragment_downloads': 1, 'logger': Quiet(), 'quiet': True,
               'cachedir': False, 'ignoreerrors': False, 'progress_hooks': [limit],
               'match_filter': lambda info, **_: ('Video exceeds 60 minutes'
                   if info.get('duration', 0) > 3600 else None)}
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(canonical, download=True)
        if not info:
            raise ClipError('Video is unavailable or outside supported limits')
        source = destination / 'source.mp4'
        metadata = probe(source)
        if not metadata['audio']:
            raise ClipError('Speech audio is required')
        retention = info.get('heatmap') or []
        saved = {'source_sha256': digest(source),
                 'provenance': 'Public replay data returned by yt-dlp for ' + canonical,
                 'heatmap': retention}
        if retention:
            (destination / 'retention.json').write_text(json.dumps(saved, indent=2) + '\n')
        return {'source': str(source), 'retention_available': bool(retention),
                'source_sha256': saved['source_sha256']}
    except Exception as error:
        if isinstance(error, ClipError):
            raise
        raise ClipError('Public source unavailable; request an upload. No access restriction was bypassed.') from None
