import cv2
import numpy as np
import math

def find_optimal_kernel(thresh_img, max_k=7, early_stop_threshold=0.05):
    """
    엘보우 기법(Elbow Method)과 ROI 샘플링을 사용하여 최적의 팽창(Dilation) 커널 크기를 계산합니다.
    """
    h, w = thresh_img.shape
    # ROI 샘플링 (정중앙 800x800)
    crop_size = min(800, h, w)
    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2
    roi = thresh_img[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    prev_count = None
    for k in range(1, max_k + 1):
        if k == 1:
            closed = roi.copy()
        else:
            kernel = np.ones((k, k), np.uint8)
            closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
            
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
        
        if prev_count is not None:
            drop_rate = (prev_count - count) / prev_count if prev_count > 0 else 0
            if drop_rate < early_stop_threshold:
                return k - 1
                
        prev_count = count
        
    return max_k
class BaseDetector:
    def process(self, image):
        raise NotImplementedError("process() 메서드를 구현해야 합니다.")

class DrawingShapeDetector(BaseDetector):
    def __init__(self, min_area=50, epsilon_factor=0.01):
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor

    def get_angle(self, pt1, pt2, pt3):
        import math
        v1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
        v2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 == 0 or mag2 == 0: return 0
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_theta = max(min(dot / (mag1 * mag2), 1.0), -1.0)
        return math.degrees(math.acos(cos_theta))

    def distance(self, pt1, pt2):
        import math
        return math.hypot(pt1[0]-pt2[0], pt1[1]-pt2[1])

    def process(self, image):
        # CAD 토폴로지 추출 결과 저장
        shapes = {
            'lines': [],          # 선분들 (점선 포함)
            'bubbles': [],        # 부품 번호 동그라미
            'rects': [],          # 핀 박스 등 네모 구멍
            'arrows': [],         # 화살표 촉
            'text_blocks': []     # 기타 텍스트 및 기호
        }
        debug = {}
        
        # 1. 흑백 변환 (벡터 스케치와 100% 동일한 원자재 필터)
        gray = np.max(image, axis=2).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        thresh_val, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # ==========================================
        # [NEW] Phase 1. 기하학 기호(핀 박스, 동그라미, 화살표 촉) 사전 필터링 및 제거
        # ==========================================
        shapes['bubbles'] = []
        shapes['rects'] = []
        shapes['arrows'] = [] # 화살표 촉 좌표 저장용
        
        symbol_mask = np.zeros_like(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(h, 1)
            
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 1. 화살표 촉(삼각형) 탐지 (면적이 작고 꼭짓점 3개)
            if len(approx) == 3 and 10 < area < 150:
                # 삼각형의 무게중심을 화살표 촉 좌표로 저장
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    shapes['arrows'].append((cx, cy))
                    cv2.drawContours(symbol_mask, [cnt], -1, 255, cv2.FILLED)
                continue
                
            if area < 50: continue
            
            # 2. 다각형 근사화 (네모 핀 박스 탐지)
            # 도면 양옆의 진짜 핀 박스는 보통 면적이 작으므로, 상한선을 500으로 엄격하게 제한!
            if len(approx) == 4 and 0.5 < aspect_ratio < 2.0 and 50 < area < 500:
                shapes['rects'].append((x, y, w, h))
                cv2.drawContours(symbol_mask, [cnt], -1, 255, cv2.FILLED)
                continue
                
            # 3. 둥근 부품 번호 기호(Bubbles) 탐지
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.75 and 100 < area < 3000:
                    shapes['bubbles'].append((x, y, w, h))
                    cv2.drawContours(symbol_mask, [cnt], -1, 255, cv2.FILLED)
                    
        # 기호 마스크 감산 (기하학 도형 통째로 삭제)
        thresh_no_symbols = cv2.bitwise_and(thresh, cv2.bitwise_not(symbol_mask))
        
        # 끊어진 결선 복원 (가로 15x1 팽창 - 핀 박스가 지워진 결선 끝단 자연스럽게 잇기)
        heal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        thresh_healed = cv2.dilate(thresh_no_symbols, heal_k)
        
        # 이후의 모든 로직은 thresh_healed(기호가 제거되고 선이 복원된 캔버스) 위에서 수행됨
        thresh = thresh_healed
        
        # ==========================================
        # [NEW] 형태학(Morphology) 연산 선행 (글자 제거 후 배선 추출)
        # ==========================================
        # 동적 하한선 계산 (글자 폰트 크기 추정)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        valid_heights = [h for h in stats[1:, cv2.CC_STAT_HEIGHT] if 5 < h < 200]
        median_h = np.median(valid_heights) if valid_heights else 30.0
        
        # 글자가 녹아 없어지도록 폰트 높이의 1.5배 길이를 열기 커널로 사용
        close_size = max(5, int(median_h * 0.5))  # 점선 이음
        open_size = max(30, int(median_h * 1.5))  # 글자 삭제, 최소 30픽셀 하한선 강제 고정!
        
        v_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_size))
        v_open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_size))
        h_close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, 1))
        h_open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, 1))
        
        v_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, v_close_k)
        v_lines_mask = cv2.morphologyEx(v_closed, cv2.MORPH_OPEN, v_open_k)
        h_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, h_close_k)
        h_lines_mask = cv2.morphologyEx(h_closed, cv2.MORPH_OPEN, h_open_k)
        
        # 글자가 완전히 녹아 없어지고 순수 뼈대만 남은 '선 레이어'
        lines_layer = cv2.bitwise_or(v_lines_mask, h_lines_mask)
        
        debug['auto_thresh_val'] = thresh_val
        debug['auto_kernel_size'] = 1 
        
        # ==========================================
        # LSD 배선 추출 (글자가 없는 lines_layer 대상)
        # ==========================================
        lsd = cv2.createLineSegmentDetector(0)
        lines_raw, _, _, _ = lsd.detect(lines_layer)

        H_lines, V_lines, D_lines = [], [], []
        floor_limit = max(30, int(median_h * 0.8))
        if lines_raw is not None:
            for line in lines_raw:
                x1, y1, x2, y2 = map(float, line.flatten())
                seg_len = self.distance((x1, y1), (x2, y2))
                if seg_len < floor_limit:
                    continue

                dx, dy = abs(x2-x1), abs(y2-y1)
                if dy <= 2 and dx > dy:
                    H_lines.append([min(x1, x2), max(x1, x2), (y1+y2)/2])
                elif dx <= 2 and dy > dx:
                    V_lines.append([min(y1, y2), max(y1, y2), (x1+x2)/2])
                else:
                    D_lines.append([x1, y1, x2, y2])

        # 점선(차폐선) 후보: lines_layer는 형태학 열기(open_size~30px 이상)로
        # 짧은 점선 조각이 통째로 지워지므로, 열기 연산 이전의 thresh(기호 제거만 된 상태)에서
        # 별도로 LSD를 돌려 짧은 조각을 수집한다.
        H_short, V_short = [], []
        lines_raw_dashed, _, _, _ = lsd.detect(thresh)
        if lines_raw_dashed is not None:
            for line in lines_raw_dashed:
                x1, y1, x2, y2 = map(float, line.flatten())
                seg_len = self.distance((x1, y1), (x2, y2))
                if seg_len >= floor_limit:
                    continue
                dx, dy = abs(x2-x1), abs(y2-y1)
                if 3 <= seg_len and dy <= 2 and dx > dy:
                    H_short.append([min(x1, x2), max(x1, x2), (y1+y2)/2])
                elif 3 <= seg_len and dx <= 2 and dy > dx:
                    V_short.append([min(y1, y2), max(y1, y2), (x1+x2)/2])

        # 2-1. 쌍선 병합 알고리즘 (중심선 축출)
        def merge_1d_lines(lines_1d, pair_gap=5):
            lines_1d.sort(key=lambda x: x[2]) # 위치(y 또는 x) 기준으로 정렬
            merged = []
            used = [False] * len(lines_1d)
            
            for i in range(len(lines_1d)):
                if used[i]: continue
                a0, a1, pa = lines_1d[i]
                cur_a0, cur_a1, cur_p = a0, a1, pa
                count = 1
                
                for j in range(i + 1, len(lines_1d)):
                    if used[j]: continue
                    b0, b1, pb = lines_1d[j]
                    
                    if pb - cur_p > pair_gap:
                        break # 정렬되어 있으므로 gap 넘어가면 종료
                        
                    ov_start, ov_end = max(cur_a0, b0), min(cur_a1, b1)
                    ov = ov_end - ov_start
                    span = min(cur_a1 - cur_a0, b1 - b0)
                    
                    if span > 0 and ov > 0.7 * span:
                        cur_a0 = min(cur_a0, b0)
                        cur_a1 = max(cur_a1, b1)
                        cur_p = (cur_p * count + pb) / (count + 1)
                        count += 1
                        used[j] = True
                        
                merged.append([cur_a0, cur_a1, cur_p])
            return merged

        def join_collinear(lines_1d, binary, axis, pos_tol=2, max_gap=8, ink_ratio=0.8):
            # 그룹핑 경계(pos_tol) 오류를 방지하기 위해 정렬 후 순차 비교
            lines_1d.sort(key=lambda s: (round(s[2] / pos_tol), s[0]))
            out = []
            
            h_img, w_img = binary.shape
            
            for s in lines_1d:
                if not out:
                    out.append(list(s))
                    continue

                m = out[-1]
                gap = s[0] - m[1]

                if abs(m[2] - s[2]) <= pos_tol and 0 <= gap <= max_gap:
                    pos = int(round((m[2] + s[2]) / 2))
                    a, b = int(m[1]), int(s[0])
                    
                    if b > a:
                        if axis == 'H':
                            pos_min, pos_max = max(0, pos-1), min(h_img, pos+2)
                            strip = binary[pos_min:pos_max, a:b].max(axis=0)
                        else:
                            pos_min, pos_max = max(0, pos-1), min(w_img, pos+2)
                            strip = binary[a:b, pos_min:pos_max].max(axis=1)
                        filled = np.mean(strip > 0)
                    else:
                        filled = 1.0

                    if filled >= ink_ratio:      # 틈새에 잉크가 80% 이상 차있음 = 교차선 = 병합
                        m[1] = max(m[1], s[1])
                        m[2] = (m[2] + s[2]) / 2
                        continue

                out.append(list(s))
            return out

        H_merged = merge_1d_lines(H_lines, pair_gap=5)
        V_merged = merge_1d_lines(V_lines, pair_gap=5)

        def merge_dashed(short_1d, pos_tol=5, min_run=3, max_seg_gap=25):
            """짧은 조각들을 같은 축/위치 기준으로 정렬해, 일정 간격으로 반복되는
            구간(점선)을 찾아 그 전체를 관통하는 하나의 선분으로 만든다.
            쌍선 병합 전이라 같은 점선이 위치 2~3px 차이로 갈라져 있을 수 있으므로
            pos_tol을 넉넉히 잡아 먼저 위치 기준 그룹을 만든다."""
            if not short_1d:
                return []
            short_1d = sorted(short_1d, key=lambda s: (s[2], s[0]))
            pos_groups, cur_pos = [], [short_1d[0]]
            for s in short_1d[1:]:
                if abs(s[2] - cur_pos[-1][2]) <= pos_tol:
                    cur_pos.append(s)
                else:
                    pos_groups.append(cur_pos)
                    cur_pos = [s]
            pos_groups.append(cur_pos)

            out = []
            for grp in pos_groups:
                grp.sort(key=lambda s: s[0])
                runs, cur = [], [grp[0]]
                for s in grp[1:]:
                    if 0 <= s[0] - cur[-1][1] <= max_seg_gap:
                        cur.append(s)
                    else:
                        runs.append(cur)
                        cur = [s]
                runs.append(cur)

                for run in runs:
                    if len(run) < min_run:
                        continue  # 조각이 몇 개 안 되면 우연/노이즈로 보고 버림
                    a0 = min(s[0] for s in run)
                    a1 = max(s[1] for s in run)
                    pos = sum(s[2] for s in run) / len(run)
                    out.append([a0, a1, pos])
            return out

        H_merged.extend(merge_dashed(H_short))
        V_merged.extend(merge_dashed(V_short))

        # 2-2. 토막 이음 (교차로에 의해 끊어진 선분 연결)
        H_joined = join_collinear(H_merged, thresh, 'H')
        V_joined = join_collinear(V_merged, thresh, 'V')
        
        pts = []
        for h in H_joined:
            pts.append((int(h[0]), int(h[2]), int(h[1]), int(h[2])))
        for v in V_joined:
            pts.append((int(v[2]), int(v[0]), int(v[2]), int(v[1])))
        for d in D_lines:
            pts.append((int(d[0]), int(d[1]), int(d[2]), int(d[3])))
            
        shapes['lines'] = pts
                    
        # 2-3. 폴리라인(Polyline) 그래프 구조화 및 경로 추출
        import collections
        def build_paths(segs, snap=6):
            node_of, nodes = {}, []
            def nid(p):
                key = (int(round(p[0]/snap)), int(round(p[1]/snap)))
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        k = (key[0]+dx, key[1]+dy)
                        if k in node_of:
                            j = node_of[k]
                            if math.hypot(p[0]-nodes[j][0], p[1]-nodes[j][1]) <= snap:
                                return j
                j = len(nodes); nodes.append(p); node_of[key] = j
                return j

            edges = []
            for (x1,y1,x2,y2) in segs:
                a, b = nid((x1,y1)), nid((x2,y2))
                if a != b:
                    edges.append((a,b))

            adj = collections.defaultdict(list)
            for i, (a, b) in enumerate(edges):
                adj[a].append((b, i))
                adj[b].append((a, i))

            used = [False]*len(edges)
            paths = []

            STRAIGHT_ANGLE_TOL = 20  # 이 각도(도) 이내로 직진하면 교차점이어도 같은 폴리라인으로 계속 이어감

            def walk(start):
                for (nxt, ei) in adj[start]:
                    if used[ei]: continue
                    used[ei] = True
                    pts = [nodes[start], nodes[nxt]]
                    cur, prev_e = nxt, ei
                    while True:
                        nxts = [(n, e) for (n, e) in adj[cur] if e != prev_e and not used[e]]
                        if not nxts: break

                        # 들어온 방향(prev -> cur)과 가장 일직선(반대 방향)에 가까운 간선을 고른다.
                        # 차수가 2든 4든(십자 교차) 상관없이, 방향이 맞는 간선이 있으면 그대로 통과한다.
                        px, py = nodes[cur][0] - pts[-2][0], nodes[cur][1] - pts[-2][1]
                        best, best_dev = None, None
                        for (n2, e2) in nxts:
                            qx, qy = nodes[n2][0] - nodes[cur][0], nodes[n2][1] - nodes[cur][1]
                            mag_p, mag_q = math.hypot(px, py), math.hypot(qx, qy)
                            if mag_p < 1e-9 or mag_q < 1e-9:
                                dev = 0
                            else:
                                cos_a = max(min((px*qx + py*qy) / (mag_p*mag_q), 1.0), -1.0)
                                dev = math.degrees(math.acos(cos_a))
                            if best_dev is None or dev < best_dev:
                                best, best_dev = (n2, e2), dev

                        if len(adj[cur]) != 2 and (best_dev is None or best_dev > STRAIGHT_ANGLE_TOL):
                            break  # 진짜 분기점(T/Y자): 직진 방향 간선이 없으면 여기서 경로를 끊는다

                        n2, e2 = best
                        used[e2] = True
                        pts.append(nodes[n2])
                        prev_e, cur = e2, n2
                    paths.append(pts)

            # 1. 단말(끝) 노드나 분기점에서 먼저 출발
            for v in list(adj.keys()):
                if len(adj[v]) != 2: 
                    walk(v)
            # 2. 남은 순환 루프(닫힌 고리) 처리
            for v in list(adj.keys()):
                walk(v)
                
            return paths
            
        def simplify(pts, tol=2.0):
            if len(pts) < 3: return pts
            out = [pts[0]]
            for i in range(1, len(pts)-1):
                ax,ay = out[-1]; bx,by = pts[i]; cx,cy = pts[i+1]
                cross = abs((bx-ax)*(cy-ay) - (by-ay)*(cx-ax))
                base = math.hypot(cx-ax, cy-ay)
                if base < 1e-9 or cross/base > tol:
                    out.append(pts[i])
            out.append(pts[-1])
            return out

        raw_paths = build_paths(pts, snap=6)
        shapes['polylines'] = [simplify(p, tol=2.0) for p in raw_paths]
        shapes['lines'] = [] # 기존 단순 라인은 숨김 처리
        
        # 3. 화살표 탐지 (V자 촉 찾기)
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                line1, line2 = pts[i], pts[j]
                p1_1, p1_2 = (line1[0], line1[1]), (line1[2], line1[3])
                p2_1, p2_2 = (line2[0], line2[1]), (line2[2], line2[3])
                
                threshold_gap = 5 
                vertex = None
                end1, end2 = None, None
                
                if self.distance(p1_1, p2_1) < threshold_gap: vertex, end1, end2 = p1_1, p1_2, p2_2
                elif self.distance(p1_1, p2_2) < threshold_gap: vertex, end1, end2 = p1_1, p1_2, p2_1
                elif self.distance(p1_2, p2_1) < threshold_gap: vertex, end1, end2 = p1_2, p1_1, p2_2
                elif self.distance(p1_2, p2_2) < threshold_gap: vertex, end1, end2 = p1_2, p1_1, p2_1
                
                if vertex:
                    angle = self.get_angle(end1, vertex, end2)
                    if 15 < angle < 75: 
                        len1 = self.distance(vertex, end1)
                        len2 = self.distance(vertex, end2)
                        # 화살표 촉은 길이가 보통 짧음
                        if 5 < len1 < 40 and 5 < len2 < 40:
                            shapes['arrows'].append(vertex)
                            
        raw_paths = build_paths(pts, snap=6)
        polylines = [simplify(p, tol=2.0) for p in raw_paths]
        
        shapes['polylines'] = []
        shapes['arrow_lines'] = []
        
        # [NEW] 2-4. 화살표 지시선 분리 (Angle Filtering)
        for poly in polylines:
            if len(poly) > 1:
                # 시작점과 끝점의 전체 기울기 검사
                dx = poly[-1][0] - poly[0][0]
                dy = poly[-1][1] - poly[0][1]
                
                # 완전히 수평(0)/수직(90)인 0이나 분모 0 에러 방지
                if dx == 0 and dy == 0: 
                    shapes['polylines'].append(poly)
                    continue
                    
                angle = math.degrees(math.atan2(abs(dy), abs(dx)))
                
                # 기울기가 15도 ~ 75도 사이인 사선(Diagonal)은 지시선으로 분류
                if 15 < angle < 75:
                    shapes['arrow_lines'].append(poly)
                else:
                    shapes['polylines'].append(poly)
                    
        shapes['lines'] = [] # 기존 단순 라인은 폴리라인으로 병합되었으므로 비움
        
        # ==========================================
        # 4. [NEW] 마스킹 & 감산 (원본 역매핑을 위한 순수 텍스트 추출)
        # ==========================================
        # 4-1. 배선(메인 폴리라인 + 지시선)이 겹친 캔버스 생성
        wire_mask = np.zeros_like(thresh)
        for poly in shapes['polylines'] + shapes['arrow_lines']:
            if len(poly) > 1:
                pts_arr = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(wire_mask, [pts_arr], isClosed=False, color=255, thickness=2)
                
        # 4-2. 원본 이진화 도면에서 선분 영역 감산
        text_raw = cv2.subtract(thresh, wire_mask)
        
        # 4-3. 찌꺼기 획 복원
        heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_healed = cv2.morphologyEx(text_raw, cv2.MORPH_CLOSE, heal_kernel)
        debug['step3_contours'] = cv2.cvtColor(text_healed, cv2.COLOR_GRAY2BGR) 
        
        # 4-4. 원본 역매핑을 위한 CCL
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_healed, connectivity=8)
        shapes['text_blocks'] = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 15: continue 
            shapes['text_blocks'].append((x, y, w, h))
                
        return {'shapes': shapes, 'debug': debug}

class CableAnalyzer:
    def __init__(self):
        self.detectors = {}

    def add_detector(self, name, detector):
        self.detectors[name] = detector

    def analyze(self, image):
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.process(image)
        return results

class Visualizer:
    COLORS = {
        'lines': (255, 105, 180),      # 메인 배선 (핫핑크)
        'arrow_lines': (0, 255, 255),  # 지시선 (노란색)
        'bubbles': (255, 165, 0),      # 주황색
        'rects': (0, 255, 0),          # 초록색
        'arrows': (0, 0, 255),         # 화살표 촉 (빨간색 과녁)
        'text_blocks': (255, 255, 255) # 텍스트 (흰색)
    }

    @staticmethod
    def draw_results(image, results):
        out_img = image.copy()
        shapes = results.get('drawing_detector', {}).get('shapes', {})
        
        for (x, y, w, h) in shapes.get('text_blocks', []):
            cv2.rectangle(out_img, (x, y), (x+w, y+h), Visualizer.COLORS['text_blocks'], 1)
            
        for (x, y, w, h) in shapes.get('rects', []):
            cv2.rectangle(out_img, (x, y), (x+w, y+h), Visualizer.COLORS['rects'], 2)
            cv2.putText(out_img, "PinBox", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Visualizer.COLORS['rects'], 1)
            
        for (x, y, w, h) in shapes.get('bubbles', []):
            cx, cy = x + w//2, y + h//2
            radius = max(w, h)//2
            cv2.circle(out_img, (cx, cy), radius, Visualizer.COLORS['bubbles'], 2)
            cv2.putText(out_img, "Bubble", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, Visualizer.COLORS['bubbles'], 1)
            
        import numpy as np
        for poly in shapes.get('polylines', []):
            if len(poly) > 1:
                pts_arr = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(out_img, [pts_arr], isClosed=False, color=Visualizer.COLORS['lines'], thickness=2)
                
        # [NEW] 지시선(사선) 그리기
        for poly in shapes.get('arrow_lines', []):
            if len(poly) > 1:
                pts_arr = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.polylines(out_img, [pts_arr], isClosed=False, color=Visualizer.COLORS['arrow_lines'], thickness=2)
                
        for tip in shapes.get('arrows', []):
            cv2.circle(out_img, tip, 15, Visualizer.COLORS['arrows'], 3)
            cv2.circle(out_img, tip, 2, Visualizer.COLORS['arrows'], cv2.FILLED) # 정중앙 점
            
        return out_img
