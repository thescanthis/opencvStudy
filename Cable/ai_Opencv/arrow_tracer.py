import cv2
import numpy as np
import math
import sys
import os

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

def detect_arrows(image_path):
    print(f"Loading {image_path}...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load {image_path}")
        return
        
    print("Binarizing...")
    gray = np.max(img, axis=2).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    print("Extracting line segments (LSD)...")
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(thresh)
    
    output = img.copy()
    arrow_tips = []
    
    if lines is not None:
        print(f"Found {len(lines)} line segments. Searching for arrow V-shapes...")
        pts = []
        for line in lines:
            x1, y1, x2, y2 = map(int, line.flatten())
            pts.append((x1, y1, x2, y2))
            
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                line1 = pts[i]
                line2 = pts[j]
                
                p1_1, p1_2 = (line1[0], line1[1]), (line1[2], line1[3])
                p2_1, p2_2 = (line2[0], line2[1]), (line2[2], line2[3])
                
                threshold = 5 # max gap between flap tips to be considered a vertex
                vertex = None
                end1 = None
                end2 = None
                
                if distance(p1_1, p2_1) < threshold: vertex, end1, end2 = p1_1, p1_2, p2_2
                elif distance(p1_1, p2_2) < threshold: vertex, end1, end2 = p1_1, p1_2, p2_1
                elif distance(p1_2, p2_1) < threshold: vertex, end1, end2 = p1_2, p1_1, p2_2
                elif distance(p1_2, p2_2) < threshold: vertex, end1, end2 = p1_2, p1_1, p2_1
                
                if vertex:
                    angle = get_angle(end1, vertex, end2)
                    if 15 < angle < 75: # Arrow head angle
                        len1 = distance(vertex, end1)
                        len2 = distance(vertex, end2)
                        if 5 < len1 < 40 and 5 < len2 < 40:
                            arrow_tips.append(vertex)
                            
    for tip in arrow_tips:
        cv2.circle(output, tip, 20, (0, 0, 255), 3) # Red circle on the tip
        
    print(f"Success! Found {len(arrow_tips)} arrow candidates.")
    
    result_path = "arrow_tracer_result.png"
    cv2.imwrite(result_path, output)
    print(f"Result saved as {result_path}")
    
    # 윈도우에서 자동으로 이미지 열기
    os.startfile(result_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # 최근 추가된 파일 중 png 파일 찾기
        files = [f for f in os.listdir('.') if f.endswith('.png') and 'result' not in f]
        if files:
            target = files[0]
        else:
            target = "Q25041761.png"
    detect_arrows(target)
