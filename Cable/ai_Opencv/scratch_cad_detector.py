import cv2
import numpy as np
import math

def get_angle(pt1, pt2, pt3):
    v1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    v2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 == 0 or mag2 == 0: return 0
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos_theta = max(min(dot / (mag1 * mag2), 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))

def distance(pt1, pt2):
    return math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

class CADTopologyDetector:
    def __init__(self):
        pass

    def process(self, image):
        results = {
            'lines': [],          # 가로/세로 실선
            'bubbles': [],        # 부품 번호 동그라미
            'rects': [],          # 커넥터 핀 네모
            'arrows': [],         # 화살표 촉 (x, y)
            'text_blocks': []     # 독립된 텍스트 덩어리
        }
        debug = {}

        # 1. 완벽한 원자재 이진화 (벡터 스케치와 100% 동일)
        gray = np.max(image, axis=2).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        thresh_val, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        debug['step2_binary'] = thresh
        
        # 2. 직선 추출 (LSD)
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(thresh)
        
        pts = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line.flatten())
                # 너무 짧은 먼지는 제외 (점선 파편은 5~10픽셀 이상)
                length = distance((x1,y1), (x2,y2))
                if length > 5:
                    results['lines'].append((x1, y1, x2, y2))
                    pts.append((x1, y1, x2, y2))

        # 3. 화살표 탐지 (V자 촉 찾기)
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                line1, line2 = pts[i], pts[j]
                
                p1_1, p1_2 = (line1[0], line1[1]), (line1[2], line1[3])
                p2_1, p2_2 = (line2[0], line2[1]), (line2[2], line2[3])
                
                threshold_gap = 5 
                vertex = None
                end1, end2 = None, None
                
                if distance(p1_1, p2_1) < threshold_gap: vertex, end1, end2 = p1_1, p1_2, p2_2
                elif distance(p1_1, p2_2) < threshold_gap: vertex, end1, end2 = p1_1, p1_2, p2_1
                elif distance(p1_2, p2_1) < threshold_gap: vertex, end1, end2 = p1_2, p1_1, p2_2
                elif distance(p1_2, p2_2) < threshold_gap: vertex, end1, end2 = p1_2, p1_1, p2_1
                
                if vertex:
                    angle = get_angle(end1, vertex, end2)
                    if 15 < angle < 75: 
                        len1 = distance(vertex, end1)
                        len2 = distance(vertex, end2)
                        if 5 < len1 < 40 and 5 < len2 < 40:
                            results['arrows'].append(vertex)
                            
        # 4. 여백(구멍)을 이용한 Bubble, Rect, Text 탐지
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w)/h
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 동그라미 (Bubble)
            if 0.7 < aspect_ratio < 1.3 and circularity > 0.75:
                results['bubbles'].append((x, y, w, h))
                continue
                
            # 네모 (Connector Pin Boxes)
            # 테두리가 곡선이라도 BoundingBox는 네모! 
            # 커넥터 핀 박스들은 보통 세로가 짧고 가로가 긺 (혹은 정사각형)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                # 속이 꽉 찬 네모 모양 구멍
                if solidity > 0.8:
                    if 0.2 < aspect_ratio < 5.0 and area > 100:
                        results['rects'].append((x, y, w, h))
                        continue
                
                # 나머지 글씨나 기호 (Text Blocks)
                if solidity < 0.8 or area <= 100:
                    results['text_blocks'].append((x, y, w, h))

        return {'shapes': results, 'debug': debug}
