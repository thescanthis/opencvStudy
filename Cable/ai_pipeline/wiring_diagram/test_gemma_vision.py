"""LM Studio(로컬 REST API, OpenAI 호환)로 로드된 Gemma 비전 모델에
핀 라벨 세로열 크롭 이미지를 보내서 점선/빗금 폰트를 얼마나 정확히
읽는지 테스트한다."""
import base64
import json
import sys
import urllib.request

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "google/gemma-4-12b-qat"


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask_gemma(image_path, prompt, max_tokens=800):
    b64 = encode_image(image_path)
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    msg = result["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    finish = result["choices"][0].get("finish_reason")
    return content, reasoning, finish


def test_single_cell():
    content, reasoning, finish = ask_gemma(
        "out/_pin_cell0.png",
        "이 이미지 박스 안에 알파벳 한 글자가 점선처럼 끊긴 획으로 그려져 있습니다. "
        "그 글자가 무엇인지 한 글자로만 답하세요.",
    )
    print("=== single cell test ===")
    print("finish_reason:", finish)
    print("content:", repr(content))
    print("reasoning (last 300 chars):", reasoning[-300:] if reasoning else None)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "single":
        test_single_cell()
        sys.exit(0)

    expected = ["S", "Q", "P", "N", "M", "L", "K", "J", "H", "G",
                "F", "E", "C", "D", "B", "A", "T", "U", "R", "V"]
    prompt = (
        "이 이미지는 케이블 배선도에서 커넥터 핀 이름을 나열한 세로 박스 목록입니다. "
        "각 박스 안에 알파벳 한 글자가 있고, 글자 획이 점선/빗금처럼 끊겨서 그려져 있습니다. "
        "위에서 아래 순서대로 각 박스 안의 알파벳을 읽어서, 쉼표로 구분된 한 줄로만 답하세요. "
        "예: S,Q,P,N,... 다른 설명은 하지 마세요."
    )
    content, reasoning, finish = ask_gemma("out/_pin_col_only.png", prompt, max_tokens=1500)
    print("finish_reason:", finish)
    print("Gemma content:")
    print(content)
    if not content and reasoning:
        print("\n(content 비어있음, reasoning 마지막 500자)")
        print(reasoning[-500:])

    letters = [c.strip().upper() for c in content.replace("\n", ",").split(",") if c.strip()]
    print(f"\nparsed: {letters}")
    print(f"expected: {expected}")

    correct = sum(1 for e, g in zip(expected, letters) if e == g)
    print(f"\naccuracy: {correct}/{len(expected)} ({correct/len(expected)*100:.1f}%)")
    for i, e in enumerate(expected):
        g = letters[i] if i < len(letters) else "?"
        mark = "OK" if e == g else "X"
        print(f"  [{mark}] expected={e} got={g}")
