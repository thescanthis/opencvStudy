"""경계선 검출(픽셀 휴리스틱) 없이, 크롭 전체를 한 번에 Gemma에게 보여주고
위에서 아래(또는 왼쪽에서 오른쪽) 순서대로 모든 박스의 글자를 한 번에
읽어달라고 요청하는 방식을 검증한다. 도면마다 노이즈/해상도가 달라 픽셀
임계값 튜닝이 매번 깨지는 문제를 피하기 위한 대안."""
import base64
import json
import urllib.request

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "google/gemma-4-12b-qat"


def ask_gemma_whole(image_path, n_expected, prompt, max_tokens=3000):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": MODEL_ID,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        LM_STUDIO_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    msg = result["choices"][0]["message"]
    return msg.get("content", "") or "", msg.get("reasoning_content", "") or ""


if __name__ == "__main__":
    expected = ["S", "Q", "P", "N", "M", "L", "K", "J", "H", "G",
                "F", "E", "C", "D", "B", "A", "T", "U", "R", "V"]
    prompt = (
        f"이 이미지는 케이블 배선도의 커넥터 핀 이름을 나열한 세로 박스 목록입니다. "
        f"총 {len(expected)}개의 박스가 위에서 아래로 있고, 각 박스 안에는 알파벳 한 "
        f"글자가 점선처럼 끊긴 획으로 그려져 있습니다. "
        f"위에서부터 아래까지 순서대로 각 박스의 글자를 읽어서, 쉼표로 구분한 "
        f"한 줄로만 답하세요. 예: S,Q,P,N,... 다른 설명은 하지 마세요."
    )
    content, reasoning = ask_gemma_whole("out/_whole_col_2x.png", len(expected), prompt)
    print("content:", repr(content))
    text = content
    if not text and reasoning:
        # content가 비어도 reasoning 안에 "S, Q, P, ..." 형태의 목록이 여러 번
        # 반복해서 나타난다(모델이 스스로 재확인하는 습관 때문). 정확히
        # n_expected개짜리 알파벳-콤마 패턴 중 마지막(가장 확신에 가까운)
        # 것을 정답으로 채택한다.
        import re
        candidates = re.findall(r'(?:[A-Z]\s*,\s*){' + str(len(expected)-1) + r'}[A-Z]', reasoning)
        if candidates:
            text = candidates[-1]
            print("(reasoning에서 회수)")

    letters = [c.strip().upper() for c in text.replace("\n", ",").split(",") if c.strip()]
    print("parsed:", letters)
    correct = sum(1 for e, g in zip(expected, letters) if e == g)
    print(f"accuracy: {correct}/{len(expected)} ({correct/len(expected)*100:.1f}%), got {len(letters)} letters")
    for i, e in enumerate(expected):
        g = letters[i] if i < len(letters) else "?"
        print(f"  [{'OK' if g==e else 'X'}] expected={e} got={g}")
