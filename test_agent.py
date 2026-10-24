import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import brief, create, revise
from media import caption_cues, srt
from selection import ClipError, Sentence, choose, sentences
from ingest import youtube_url, download


def words(text='One clear sentence. Another complete thought. Unfinished tail'):
    return [{'start': i, 'end': i + .7, 'text': t} for i, t in enumerate(text.split())]


class BoundaryTests(unittest.TestCase):
    def test_sentence_groups_and_tail(self):
        result = sentences(words(), 20)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].text, 'Another complete thought.')
        self.assertLessEqual(result[1].end, 6)

    def test_padding_never_slices_adjacent_words(self):
        w = [{'start': 0, 'end': 1, 'text': 'First.'},
             {'start': 1.01, 'end': 2, 'text': 'Second.'}]
        result = sentences(w, 2.01)
        self.assertLessEqual(result[0].end, w[1]['start'])
        self.assertGreaterEqual(result[1].start, w[0]['end'])
        self.assertLessEqual(result[0].end, result[1].start)

    def test_overlapping_sentence_times_rejected(self):
        with self.assertRaises(ClipError):
            sentences([{'start': 0, 'end': 2, 'text': 'First.'},
                       {'start': 1.5, 'end': 3, 'text': 'Second.'}], 4)

    def test_incomplete_speech_rejected(self):
        with self.assertRaises(ClipError):
            sentences(words('no ending punctuation'), 10)

    def test_invalid_timestamps_rejected(self):
        for bad in [float('nan'), float('inf'), True, -1, 500]:
            w = words()
            w[0]['start'] = bad
            with self.subTest(bad=bad), self.assertRaises(ClipError):
                sentences(w, 20)

    def test_nonmonotonic_times_rejected(self):
        w = words()
        w[3]['start'] = .1
        with self.assertRaises(ClipError):
            sentences(w, 20)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.items = [Sentence(i * 10, i * 10 + 9,
                               'Unique lesson %d complete.' % i,
                               ({'start': i * 10, 'end': i * 10 + 9, 'text': 'Lesson.'},))
                      for i in range(8)]

    def test_exact_count_nonoverlapping(self):
        value = choose(self.items, 3, 9, 29)
        self.assertEqual(len(value), 3)
        self.assertTrue(all(a['end'] <= b['start'] for a, b in zip(value, value[1:])))

    def test_retention_data_dominates_lexical_score(self):
        value = choose(self.items, 1, 9, 9,
                       heatmap=[{'start_time': 41, 'end_time': 45, 'value': .98}])
        self.assertEqual(value[0]['start'], 40)
        self.assertEqual(value[0]['selection_mode'], 'retention')
        self.assertIn('.980', value[0]['reason'])

    def test_missing_data_is_disclosed(self):
        value = choose(self.items, 1, 9, 19)
        self.assertEqual(value[0]['selection_mode'], 'content_analysis_heuristic')
        self.assertIn('No replay/retention data', value[0]['reason'])

    def test_insufficient_footage_is_explicit(self):
        with self.assertRaisesRegex(ClipError, 'Only 2'):
            choose(self.items[:2], 3, 9, 9)

    def test_invalid_heatmap_is_rejected(self):
        for heatmap in [[None], [{'start_time': 3, 'end_time': 2, 'value': 1}],
                        [{'start_time': 1, 'end_time': 2, 'value': float('nan')}]]:
            with self.subTest(heatmap=heatmap), self.assertRaises(ClipError):
                choose(self.items, 1, 9, 9, heatmap=heatmap)


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.request = {'rights_confirmed': True, 'clip_count': 3,
                        'formats': ['vertical', 'square'], 'captions': True}

    def test_rights_are_required(self):
        self.request['rights_confirmed'] = 'yes'
        with self.assertRaises(ClipError):
            brief(self.request)

    def test_unsupported_formats_stop_entire_request(self):
        self.request['formats'].append('unknown')
        with self.assertRaisesRegex(ClipError, 'none may be omitted'):
            brief(self.request)

    def test_caption_choice_is_required(self):
        del self.request['captions']
        with self.assertRaises(ClipError):
            brief(self.request)

    def test_duplicate_formats_rejected(self):
        self.request['formats'] = ['square', 'square']
        with self.assertRaises(ClipError):
            brief(self.request)

    def test_no_overwrite_before_media_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ClipError, 'exists'):
                create('unused', self.request, folder)


class CaptionTests(unittest.TestCase):
    def test_cues_use_actual_words_and_relative_times(self):
        w = words('A complete sentence. Another sentence.')
        cues = caption_cues(w, 0, 5)
        self.assertEqual(' '.join(c['text'] for c in cues), 'A complete sentence. Another sentence.')
        self.assertTrue(all(c['start'] < c['end'] <= 5 for c in cues))
        self.assertIn('00:00:00,000', srt(cues))

    def test_midword_cut_is_rejected(self):
        with self.assertRaisesRegex(ClipError, 'cuts through'):
            caption_cues(words(), .3, 9)


class InputLinkTests(unittest.TestCase):
    def test_only_video_identifier_survives_normalization(self):
        self.assertEqual(youtube_url('https://youtu.be/abcdefghijk?tracking=ignored'),
                         'https://www.youtube.com/watch?v=abcdefghijk')

    def test_private_and_unrelated_destinations_are_rejected(self):
        for url in ['http://youtube.com/watch?v=abcdefghijk',
                    'https://youtube.com.evil.example/watch?v=abcdefghijk',
                    'https://127.0.0.1/video.mp4',
                    'https://youtube.com:9000/watch?v=abcdefghijk',
                    'https://user:password@youtube.com/watch?v=abcdefghijk',
                    'https://youtube.com/playlist?list=abcdefghijk']:
            with self.subTest(url=url), self.assertRaises(ClipError):
                youtube_url(url)

    def test_link_download_requires_rights_before_network(self):
        with self.assertRaisesRegex(ClipError, 'rights'):
            download('https://youtu.be/abcdefghijk', 'unused-test-folder')


if __name__ == '__main__':
    unittest.main()
