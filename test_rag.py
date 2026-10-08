"""API 비용 없이 실행하는 검색 안전장치 검사."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock
from langchain_core.documents import Document
import app


class RetrievalTests(unittest.TestCase):
    def test_text_file_encodings(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'note.md'
            for encoding in ('utf-8-sig', 'utf-16', 'cp949'):
                path.write_text('한글 문서', encoding=encoding)
                self.assertEqual(app.read_text_file(path), '한글 문서')

    def test_amount_and_country_are_read_from_source(self):
        texts = [
            '여비지급구분표\n구분: 제1호 | 해당공무원: 1급 공무원\n구분: 제2호 | 해당공무원: 제1호에 해당하지 않는 공무원',
            '국외 여비 지급표\n(단위 : 미 달러화($))\n2. 국가 및 도시별 등급 구분\n다. 다등급\n테스트국, 다른국\n'
            '[표 셀 구조:]\n구분: 별표 1의 제2호에 해당하는 사람 | 등급: 다 | 숙박비: 실비(상한액: 246)',
        ]
        blocks = {f'E{i}': Document(page_content=t, metadata={'source': 'test.pdf', 'page': i}) for i, t in enumerate(texts)}
        result = app.resolve_table_answer('7급 공무원 테스트국 숙박비', blocks, 'overseas')
        self.assertIsNotNone(result)
        self.assertIn('246', result['answer'])
        self.assertIn('다등급', result['answer'])

    def test_table_rows_keep_correspondence(self):
        rows = app.table_rows([['구분', '등급', '금액'], ['유형A', '가\n나', '11\n22'], [None, '다', '33']])
        self.assertIn('구분: 유형A | 등급: 나 | 금액: 22', rows)
        self.assertIn('구분: 유형A | 등급: 다 | 금액: 33', rows)

    def test_toc_does_not_remove_body(self):
        self.assertTrue(app.is_toc('\n'.join(f'제목······ {i}' for i in range(6))))
        self.assertFalse(app.is_toc('목차에 따른 실제 규정\n숙박비 지급표\n제1호 실비\n제2호 상한액'))

    def test_korean_spacing(self):
        self.assertIn('숙박비', app.tokenize('숙 박 비'))
        self.assertIn('이집트', app.tokenize('이집트에 출장'))

    def test_fake_evidence_is_rejected(self):
        draft = app.Draft(status='answered', answer='답변', evidence_ids=['missing'], missing=[], followup_queries=[])
        self.assertFalse(app.validate_ids(draft, {'real': Document(page_content='원문')}))
        draft.evidence_ids = []
        self.assertFalse(app.validate_ids(draft, {}))

    def test_context_limits_and_neighbours(self):
        pages = {str(i): Document(page_content='원문' * 100, metadata={'source': 'a.pdf', 'page': i, 'pid': str(i)})
                 for i in range(1, 40)}
        index = app.SearchIndex(None, None, [], pages, [])
        with patch('app.hybrid_search', return_value=['10']):
            selected, _ = app.retrieve(index, ['질문'], [])
        self.assertEqual(set(selected), {'9', '10', '11'})
        with patch('app.hybrid_search', return_value=list(pages)):
            selected, _ = app.retrieve(index, ['질문'] * 8, list(pages))
        self.assertLessEqual(len(selected), app.MAX_PAGES)
        self.assertLessEqual(sum(len(pages[p].page_content) for p in selected), app.MAX_CONTEXT_CHARS)

    def test_audit_failure_never_displays_draft(self):
        doc = Document(page_content='원문', metadata={'source': 'a.pdf', 'page': 1})
        index = app.SearchIndex(None, None, [], {'1': doc}, [])
        responses = [app.SearchPlan(queries=[]), app.Draft(status='answered', answer='틀린 금액 999',
                     evidence_ids=['E1'], missing=[], followup_queries=[]),
                     app.Audit(supported=False, reason='금액 불일치', followup_queries=[]),
                     app.Draft(status='answered', answer='틀린 금액 999',
                               evidence_ids=['E1'], missing=[], followup_queries=[]),
                     app.Audit(supported=False, reason='금액 불일치', followup_queries=[])]
        chains = [Mock(invoke=Mock(return_value=r)) for r in responses]
        with patch('app.model_chain', side_effect=chains), patch('app.retrieve', return_value=(['1'], [])):
            result = app.answer_question('질문', index, 'unused')
        self.assertEqual(result['status'], 'citation_failed')
        self.assertNotIn('999', result['answer'])

    def test_invalid_evidence_id_is_repaired_once(self):
        doc = Document(page_content='원문 금액 20', metadata={'source': 'a.pdf', 'page': 1})
        blocks = {'E1': doc}
        wrong = app.Draft(status='answered', answer='20', evidence_ids=['E999'], missing=[], followup_queries=[])
        corrected = app.Draft(status='answered', answer='20', evidence_ids=['E1'], missing=[], followup_queries=[])
        trace = {'drafts': [], 'audit': []}
        with patch('app.model_chain', side_effect=[Mock(invoke=Mock(return_value=wrong)),
                                                  Mock(invoke=Mock(return_value=corrected))]):
            result = app.draft_with_valid_ids('금액?', blocks, 'unused', '', trace)
        self.assertEqual(result.evidence_ids, ['E1'])
        self.assertEqual(trace['audit'], ['invalid_evidence_id'])

    def test_direct_policy_quotes_source_amount(self):
        text = ('3)친지 집 등에 숙박하거나 2인 이상이 공동으로 숙박한 경우\n'
                '가) 숙박을 필요로 하는 공무상 여행 시 친지 집 등에서 숙박하여 숙박비를 지출하지 않은 경우 '
                '출장 후 정산 신청을 하는 경우 1야당 23,000원을 지급할 수 있다.')
        doc = Document(page_content=text, metadata={'source': 'rules.pdf', 'page': 18})
        result = app.resolve_direct_policy_answer('친척집 숙박비 얼마야?', {'E1': doc}, 'domestic')
        self.assertEqual(result['status'], 'answered')
        self.assertIn('23,000원', result['answer'])
        self.assertEqual(result['ids'], ['E1'])

    def test_domestic_limit_needs_pay_category(self):
        text = ('[별표 2] 국내 여비 지급표\n'
                '구분: 제1호 | 숙박비(1박당): 실비\n'
                '구분: 제2호 | 숙박비(1박당): 실비 (상한액: 지역별)')
        doc = Document(page_content=text, metadata={'source': 'rules.pdf', 'page': 84})
        result = app.resolve_direct_policy_answer('통영 숙박비 상한액 얼마야?', {'E1': doc}, 'domestic')
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['ids'], ['E1'])

    def test_special_fixed_rate_is_not_domestic_limit_table(self):
        special = Document(page_content=(
            '[별표 2] 국내 여비 지급표를 참조하는 예외 정액여비\n'
            '구분: 제1호 | 숙박비: 77,000원\n'
            '구분: 제2호 | 숙박비: 55,000원'), metadata={'source': 'rules.pdf', 'page': 20})
        standard = Document(page_content=(
            '[별표 2] 국내 여비 지급표\n'
            '구분: 제1호 | 숙박비(1박당): 실비\n'
            '구분: 제2호 | 숙박비(1박당): 실비 (상한액: 지역별)'),
            metadata={'source': 'rules.pdf', 'page': 84})
        result = app.resolve_direct_policy_answer('통영 숙박비 상한액 얼마야?',
                                                  {'E20': special, 'E84': standard}, 'domestic')
        self.assertEqual(result['ids'], ['E84'])

    def test_audit_revision_uses_only_verified_answer(self):
        doc = Document(page_content='원문 금액 20', metadata={'source': 'a.pdf', 'page': 1})
        index = app.SearchIndex(None, None, [], {'1': doc}, [])
        bad = app.Draft(status='answered', answer='30', evidence_ids=['E1'], missing=[], followup_queries=[])
        good = app.Draft(status='answered', answer='20', evidence_ids=['E1'], missing=[], followup_queries=[])
        responses = [app.SearchPlan(queries=[]), bad,
                     app.Audit(supported=False, reason='금액 불일치', followup_queries=[]), good,
                     app.Audit(supported=True, reason='원문과 일치', followup_queries=[])]
        chains = [Mock(invoke=Mock(return_value=r)) for r in responses]
        with patch('app.model_chain', side_effect=chains), patch('app.retrieve', return_value=(['1'], [])):
            result = app.answer_question('금액?', index, 'unused')
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['answer'], '20')
        self.assertEqual(result['citations'][0]['page'], 1)


if __name__ == '__main__':
    unittest.main()
