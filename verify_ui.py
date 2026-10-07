"""Streamlit 화면의 실제 실행과 출처 표시를 확인합니다(API 비용 발생)."""
from streamlit.testing.v1 import AppTest


def main():
    app = AppTest.from_file('app.py').run(timeout=180)
    assert not app.exception and not app.error, '초기 화면 실행 실패'
    app.chat_input[0].set_value('9급 공무원 이집트 숙박비 알려줘').run(timeout=180)
    assert not app.exception and not app.error, '대화 화면 실행 실패'
    assert any('137' in item.value for item in app.markdown), '답변 금액 미표시'
    assert len(app.code) >= 3, '세 원문 근거 미표시'
    assert all(item.value for item in app.code), '빈 근거'
    app.checkbox[0].check().run(timeout=180)
    assert not app.exception and not app.error
    assert len(app.json) >= 1, '진단 화면 미표시'
    print('STREAMLIT_UI_PASSED', flush=True)


if __name__ == '__main__':
    main()
