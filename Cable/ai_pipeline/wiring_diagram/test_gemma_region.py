"""도면 전체 페이지 이미지에서 '배선도(회로도)' 영역의 bounding box를
Gemma에게 물어서, 크롭을 자동화할 수 있는지 검증한다."""
import base64
import json
import re
import urllib.request

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "google/gemma-4-12b-qat"

PROMPT = (
    "이 이미지는 기계/케이블 도면 전체 페이지입니다. "
    "이 페이지 안에는 여러 영역이 있는데, 그 중 '배선도'(또는 회로도, wiring "
    "diagram)라고 표시된 부분, 즉 커넥터/핀 번호를 선으로 연결해서 그린 "
    "회로도 다이어그램 영역을 찾아주세요. 치수선, 부품목록 표, 수정내역 표, "
    "타이틀블록은 제외하고, 순수하게 배선 연결을 그린 다이어그램 영역만 "
    "찾으세요. 그 영역의 bounding box를 이미지 전체 기준 퍼센트(0~100)로 "
    "다음 JSON 형식으로만 답하세요, 다른 설명 없이:\n"
    '{"x_min": 숫자, "y_min": 숫자, "x_max": 숫자, "y_max": 숫자}'
)


def ask_gemma_region(image_path, prompt=PROMPT, max_tokens=1500):
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


def extract_json_obj(text):
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


if __name__ == "__main__":
    content, reasoning = ask_gemma_region("out/_a60023104_full_small.png")
    print("content:", repr(content[:1000]))

    data = extract_json_obj(content)
    if data is None and reasoning:
        print("(content 파싱 실패, reasoning에서 재시도)")
        data = extract_json_obj(reasoning)

    if data is None:
        print("파싱 완전 실패")
        print("reasoning tail:", reasoning[-1500:] if reasoning else None)
    else:
        print("검출된 bbox(%):", data)
        # 실제 픽셀로 변환해서 크롭 검증
        import cv2
        img = cv2.imread("out/_a60023104_full.png")
        h, w = img.shape[:2]
        x0 = int(data["x_min"] / 100 * w)
        y0 = int(data["y_min"] / 100 * h)
        x1 = int(data["x_max"] / 100 * w)
        y1 = int(data["y_max"] / 100 * h)
        print(f"픽셀 좌표: ({x0},{y0}) - ({x1},{y1})")
        crop = img[y0:y1, x0:x1]
        cv2.imwrite("out/_auto_region_crop.png", crop)
        print("저장: out/_auto_region_crop.png", crop.shape)
