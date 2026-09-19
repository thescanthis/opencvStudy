"""박스/리스트 형태를 가정하지 않고, 회로도 스타일 배선도에서도 동작하는지
검증한다. 개수를 세거나 등분할하지 않고, Gemma에게 이미지 안의 모든 텍스트
라벨을 좌표(%)와 함께 JSON으로 뽑아달라고 요청한다."""
import base64
import json
import re
import urllib.request

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "google/gemma-4-12b-qat"

PROMPT = (
    "이 이미지는 케이블 배선도(회로도) 도면입니다. "
    "이 안에 보이는 모든 텍스트 라벨(예: P1, P2 같은 커넥터 이름, "
    "49, 10, 47 같은 핀/전선 번호, 원 안의 풍선번호 숫자)을 찾아서 "
    "JSON 배열로만 답하세요. 도형(원, 삼각형, 반원, 화살표, 선)은 무시하고 "
    "텍스트만 추출하세요. 각 항목의 형식:\n"
    '{"text": "읽은 글자", "x": 이미지 가로 기준 왼쪽에서부터의 위치(0~100 사이 숫자, %), '
    '"y": 이미지 세로 기준 위에서부터의 위치(0~100 사이 숫자, %)}\n'
    "다른 설명 없이 JSON 배열만 출력하세요."
)


def ask_gemma_coords(image_path, prompt=PROMPT, max_tokens=3000):
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


def extract_json_array(text):
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    expected = ["P1", "49", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
                "10", "10", "10", "10", "10", "10",
                "17", "18", "1", "1", "47", "P9", "10", "P10", "19", "47", "P11", "1"]

    content, reasoning = ask_gemma_coords("out/_a60023104_wiring_small.png")
    print("content:", repr(content[:2000]))

    data = extract_json_array(content)
    if data is None and reasoning:
        print("(content에서 파싱 실패, reasoning에서 재시도)")
        data = extract_json_array(reasoning)

    if data is None:
        print("파싱 완전 실패")
        print("reasoning tail:", reasoning[-1000:] if reasoning else None)
    else:
        print(f"\n검출된 항목 수: {len(data)} (기대: {len(expected)})")
        for item in data:
            print(f"  text={item.get('text')!r} x={item.get('x')} y={item.get('y')}")

        got_texts = [str(item.get("text", "")).strip() for item in data]
        found_expected = [e for e in expected if e in got_texts]
        print(f"\n기대값 중 검출된 것: {len(found_expected)}/{len(expected)}")
        missing = [e for e in expected if e not in got_texts]
        print("놓친 것:", missing)
        extra = [t for t in got_texts if t not in expected]
        print("엉뚱하게 추가된 것:", extra)
