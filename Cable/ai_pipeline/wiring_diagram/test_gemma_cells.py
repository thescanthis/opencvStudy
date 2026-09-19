"""핀 이름 20개 셀을 각각 개별로 Gemma에 물어서 정확도를 측정한다.
한 번에 세로열 전체를 묻는 것보다 셀 단위가 더 안정적인지 검증."""
from test_gemma_vision import ask_gemma

expected = ["S", "Q", "P", "N", "M", "L", "K", "J", "H", "G",
            "F", "E", "C", "D", "B", "A", "T", "U", "R", "V"]

PROMPT = (
    "이 이미지 박스 안에 알파벳 한 글자가 점선처럼 끊긴 획으로 그려져 있습니다. "
    "그 글자가 무엇인지 한 글자로만 답하세요."
)

if __name__ == "__main__":
    correct = 0
    results = []
    for i, exp in enumerate(expected):
        path = f"out/_cell_{i:02d}.png"
        content, reasoning, finish = ask_gemma(path, PROMPT, max_tokens=400)
        got = content.strip().upper()[:1] if content.strip() else "?"
        ok = (got == exp)
        correct += ok
        results.append((exp, got, ok))
        print(f"[{i:02d}] expected={exp} got={got!r} {'OK' if ok else 'X'}")

    print(f"\naccuracy: {correct}/{len(expected)} ({correct/len(expected)*100:.1f}%)")
