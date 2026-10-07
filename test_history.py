"""실제 대화를 건드리지 않고 임시 폴더에서 저장·복원·초기화를 검사합니다."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import app


class HistoryTests(TestCase):
    def test_refresh_code_change_and_reset(self):
        result = {'answer': '테스트 답변', 'citations': [], 'status': 'answered', 'debug': {}}
        with TemporaryDirectory() as folder, \
             patch.object(app, 'HISTORY_PATH', Path(folder) / 'messages.json'), \
             patch.object(app, 'list_data_files', return_value=[]), \
             patch.object(app, 'data_fingerprint', return_value='version1') as fingerprint, \
             patch.object(app, 'build_index', return_value=None), \
             patch.object(app, 'answer_question', return_value=result), \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'}):
            at = AppTest.from_string('import app; app.main()').run()
            at.chat_input[0].set_value('보존할 질문').run()
            self.assertFalse(at.exception)
            self.assertEqual(len(app.load_history()), 2)
            fingerprint.return_value = 'version2'
            at.run()
            self.assertEqual(len(at.chat_message), 2)
            # 새 세션은 브라우저 새로고침/서버 재시작 후의 메모리 없는 상태를 재현합니다.
            fresh = AppTest.from_string('import app; app.main()').run()
            self.assertFalse(fresh.exception)
            self.assertEqual(len(fresh.chat_message), 2)
            fresh.button(key='reset_history').click().run()
            self.assertFalse(fresh.exception)
            self.assertEqual(app.load_history(), [])
            self.assertEqual(len(fresh.chat_message), 0)
            reopened = AppTest.from_string('import app; app.main()').run()
            self.assertEqual(len(reopened.chat_message), 0)

    def test_corrupted_file_is_preserved(self):
        with TemporaryDirectory() as folder, patch.object(app, 'HISTORY_PATH', Path(folder) / 'messages.json'):
            app.HISTORY_PATH.write_text('broken', encoding='utf-8')
            with self.assertRaises(ValueError):
                app.load_history()
            self.assertEqual(app.HISTORY_PATH.read_text(encoding='utf-8'), 'broken')


if __name__ == '__main__':
    main()
