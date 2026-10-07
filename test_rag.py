"""API 비용 없이 실행하는 검색 안전장치 검사."""
import unittest
from unittest.mock import patch, Mock
from langchain_core.documents import Document
import app


class RetrievalTests(unittest.TestCase):
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
                     app.Audit(supported=False, reason='금액 불일치', followup_queries=[])]
        chains = [Mock(invoke=Mock(return_value=r)) for r in responses]
        with patch('app.model_chain', side_effect=chains), patch('app.retrieve', return_value=(['1'], [])):
            result = app.answer_question('질문', index, 'unused')
        self.assertEqual(result['status'], 'citation_failed')
        self.assertNotIn('999', result['answer'])


if __name__ == '__main__':
    unittest.main()
