"""실제 API를 사용하는 회귀 검증: uv run python -W error verify_rag.py."""
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import app


def main():
    load_dotenv(app.ROOT / '.env')
    key = os.environ['OPENAI_API_KEY']
    index = app.build_index(app.data_fingerprint(app.list_data_files()), hashlib.sha256(key.encode()).hexdigest(), key)
    questions = [
        '9급 공무원 이집트 숙박비 알려줘',
        '근무지내 교육파견 1일 여비 알려줘',
        '2급 공무원 부산 숙박비 얼마야?',
        '초콜릿 케이크 만드는 레시피 알려줘',
    ]
    results = []
    for question in questions:
        result = app.answer_question(question, index, key)
        results.append({'question': question, **result})
        print(json.dumps({'question': question, 'status': result['status'], 'answer': result['answer'],
                          'citations': [(c['source'], c['page']) for c in result['citations']],
                          'audit': result['debug']['audit']}, ensure_ascii=False), flush=True)
    # 키나 요청 헤더는 저장하지 않고 문서와 답변 대조에 필요한 결과만 보관합니다.
    Path('rag_verification.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    assert results[0]['status'] == 'answered'
    assert '137' in results[0]['answer'] and '상한' in results[0]['answer']
    assert len(results[0]['citations']) >= 3
    assert results[1]['status'] == 'insufficient'
    assert '인재개발' in results[1]['answer']
    assert results[2]['status'] == 'answered'
    assert '실비' in results[2]['answer'] and '제1호' in results[2]['answer']
    assert results[3]['status'] in {'insufficient', 'refused'}
    assert not results[3]['citations']
    print('FOUR_QUESTION_CHECKS_PASSED', flush=True)


if __name__ == '__main__':
    main()
