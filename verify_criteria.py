"""화면에 제시된 7가지 RAG 동작 기준을 실제 OpenAI API로 검증합니다."""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError

import app


CASES = [
    {"type": "정상·단일 근거", "question": "근무지 외 국내출장 시 지급되는 여비 항목은 무엇인가요?"},
    {"type": "조건형·복합 상황", "question": "서울 거주·세종 근무자가 서울에서 대구로 바로 출장 가면 운임은 어떻게 지급되나요?"},
    {"type": "금액·계산형", "question": "제주 2박 3일 출장에서 숙박비가 5만2천 원, 4만7천 원이면 얼마를 받을 수 있나요?"},
    {"type": "예외·경계 사례", "question": "기상 악화로 출장 기간이 하루 늘어난 경우 여비를 추가 지급할 수 있나요?"},
    {"type": "정보 부족 질문", "question": "제주 출장 숙박비는 얼마인가요?"},
    {"type": "문서 외·답변 불가", "question": "민간기업 직원 출장비 기준도 알려주세요."},
    {"type": "오답 유도·허위 전제", "question": "자가용 출장도 일비를 무조건 절반으로 깎나요?"},
]


def offline_source_checks() -> list[dict]:
    """API 없이 PDF 원문에 각 평가 기준의 정답 근거가 존재하는지 확인합니다."""
    documents = app.load_documents(app.list_data_files())
    pages = {(d.metadata["source"], d.metadata["page"]): app.compact_text(d.page_content) for d in documents}

    def page(source_prefix: str, number: int) -> str:
        matches = [text for (source, page_number), text in pages.items()
                   if source.startswith(source_prefix) and page_number == number]
        if len(matches) != 1:
            raise AssertionError(f"원문 페이지를 유일하게 찾지 못함: {source_prefix} {number}")
        return matches[0]

    checks = [
        ("정상·단일 근거", all(word in page("2024", 43) for word in ["운임", "숙박비", "식비", "일비"]),
         "처리기준 PDF 43쪽의 근무지 외 여비 4개 항목"),
        ("조건형·복합 상황", all(word in page("2024", 11) for word in ["거주지", "목적지", "초과하지못한다"]),
         "처리기준 PDF 11쪽의 거주지 직접 출발 및 근무지 기준 상한"),
        ("금액·계산형", all(word in page("공무원여비100", 31) for word in ["5만2천원", "4만7천원", "9만9천원", "10만원"]),
         "100문100답 PDF 31쪽의 Q&A 48 계산"),
        ("예외·경계 사례", all(word in page("공무원여비100", 31) for word in ["기상악화", "추가지급"]),
         "100문100답 PDF 31쪽의 Q&A 50 예외"),
        ("정보 부족 질문", all(word in page("2024", 84) for word in ["제1호", "제2호", "실비", "상한액"]),
         "처리기준 PDF 84쪽에서 지급구분별 기준이 달라 추가 조건 필요"),
        ("문서 외·답변 불가", not any("민간기업직원출장비기준" in text for text in pages.values()),
         "두 PDF에 민간기업 직원 출장비 기준 없음"),
        ("오답 유도·허위 전제", "자가용이용시일비는감액하지아니함" in page("공무원여비100", 25),
         "100문100답 PDF 25쪽의 Q&A 33"),
    ]
    return [{"type": name, "passed": passed, "evidence": evidence} for name, passed, evidence in checks]


def contains_all(text: str, words: list[str]) -> bool:
    compact = app.compact_text(text)
    return all(app.compact_text(word) in compact for word in words)


def evaluate(case_number: int, result: dict) -> tuple[bool, list[str]]:
    """답변 내용과 실제 출처 페이지를 함께 검사합니다."""
    answer = result["answer"]
    citations = {(c["source"], c["page"]) for c in result["citations"]}
    reasons = []

    if case_number == 1:
        ok = contains_all(answer, ["운임", "숙박비", "식비", "일비"])
        ok &= any(page in {8, 43} and source.startswith("2024") for source, page in citations)
        reasons.append("4개 항목과 처리기준 원문 출처 필요")
    elif case_number == 2:
        ok = contains_all(answer, ["서울", "대구", "세종", "초과하지"])
        ok &= any(page == 11 and source.startswith("2024") for source, page in citations)
        reasons.append("서울→대구 운임, 세종→대구 상한, 영 제6조 근거 필요")
    elif case_number == 3:
        ok = contains_all(answer, ["9만9천", "10만", "실제"])
        ok &= any(page == 31 and source.startswith("공무원여비100") for source, page in citations)
        reasons.append("52,000+47,000=99,000원과 2박 상한 100,000원 근거 필요")
    elif case_number == 4:
        ok = contains_all(answer, ["추가", "숙박비", "식비", "일비"])
        ok &= any(page in {11, 31} for _, page in citations)
        reasons.append("천재지변·부득이한 사유와 추가 여비 가능 근거 필요")
    elif case_number == 5:
        required_conditions = any(word in answer for word in ["직급", "제1호", "제2호"])
        required_conditions &= any(word in answer for word in ["숙박 일수", "몇 박", "1박", "실제 지출"])
        ok = result["status"] == "insufficient" and required_conditions
        reasons.append("직급/지급구분과 숙박일수 또는 실제 지출액을 되물어야 함")
    elif case_number == 6:
        ok = result["status"] in {"insufficient", "refused"}
        ok &= not result["citations"] and any(word in answer for word in ["민간", "문서", "자료"])
        reasons.append("공무원 자료의 범위를 안내하고 민간 기준을 추측하지 않아야 함")
    else:
        compact = app.compact_text(answer)
        ok = "감액하지" in compact or ("절반" in compact and any(word in answer for word in ["아니", "않"]))
        ok &= any(page == 25 and source.startswith("공무원여비100") for source, page in citations)
        reasons.append("자가용은 공용·임차 차량이 아니므로 일비를 감액하지 않는다는 근거 필요")

    if result["status"] in {"api_error", "processing_error", "citation_failed"}:
        ok = False
        reasons.append(f"처리 상태가 {result['status']}")
    return bool(ok), reasons


def main() -> None:
    load_dotenv(app.ROOT / ".env")
    api_key = os.environ["OPENAI_API_KEY"]
    offline = offline_source_checks()
    Path("rag_criteria_offline_report.json").write_text(
        json.dumps(offline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not all(item["passed"] for item in offline):
        print(json.dumps(offline, ensure_ascii=False), flush=True)
        raise SystemExit(1)

    files = app.list_data_files()
    try:
        index = app.build_index(app.data_fingerprint(files), hashlib.sha256(api_key.encode()).hexdigest(), api_key)
    except OpenAIError as error:
        code = getattr(error, "code", None) or type(error).__name__
        print(f"END_TO_END_BLOCKED={code}", flush=True)
        print("OFFLINE_SOURCE_CHECKS=7/7", flush=True)
        raise SystemExit(2)

    report = []
    for number, case in enumerate(CASES, start=1):
        result = app.answer_question(case["question"], index, api_key)
        passed, expectations = evaluate(number, result)
        entry = {
            **case,
            "passed": passed,
            "status": result["status"],
            "answer": result["answer"],
            "citations": [{"source": c["source"], "page": c["page"], "id": c["id"]} for c in result["citations"]],
            "expectations": expectations,
            "audit": result["debug"].get("audit", []),
        }
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    Path("rag_criteria_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    passed_count = sum(item["passed"] for item in report)
    print(f"RESULT={passed_count}/{len(report)}", flush=True)
    if passed_count != len(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
