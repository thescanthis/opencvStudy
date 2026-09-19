import os
import numpy as np
import trimesh
from PIL import Image
from skimage.morphology import skeletonize
import networkx as nx
from scipy.interpolate import splprep, splev
from abc import ABC, abstractmethod


class GeometryBuilderBase(ABC):
    @abstractmethod
    def build(self) -> trimesh.Trimesh:
        pass


def extract_3d_centerline(mask_map: np.ndarray, depth_map: np.ndarray, depth_scale: float) -> np.ndarray:
    """
    마스크에서 케이블의 중심선(Skeleton)을 추출하고 Depth값을 입혀 부드러운 3D 곡선을 반환합니다.
    """
    h, w = mask_map.shape
    
    # 1. 스켈레톤 추출
    binary_mask = mask_map > 0.5
    skeleton = skeletonize(binary_mask)
    y_idx, x_idx = np.nonzero(skeleton)
    
    if len(y_idx) < 10:
        # 픽셀이 너무 적으면 그냥 직선 반환 (Fallback)
        return np.array([[-w/4, 0, 0], [w/4, 0, 0]])

    # 2. 픽셀들을 그래프로 연결 (8방향)
    G = nx.Graph()
    points = list(zip(x_idx, y_idx))
    pt_dict = {pt: i for i, pt in enumerate(points)}
    
    for x, y in points:
        u = pt_dict[(x, y)]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx_pt = (x + dx, y + dy)
                if nx_pt in pt_dict:
                    v = pt_dict[nx_pt]
                    G.add_edge(u, v, weight=np.sqrt(dx**2 + dy**2))
                    
    # 3. 그래프에서 가장 긴 경로(주요 케이블 라인) 찾기
    # 모든 컴포넌트 중 가장 큰 서브그래프 선택
    largest_cc = max(nx.connected_components(G), key=len)
    sub_G = G.subgraph(largest_cc)
    
    # 단말 노드(Degree==1) 찾기
    endpoints = [n for n, d in sub_G.degree() if d == 1]
    if len(endpoints) < 2:
        endpoints = list(sub_G.nodes())[:2]
        
    # 두 단말 노드 사이의 최단 경로 중 가장 긴 것 선택 (케이블 양 끝단)
    longest_path = []
    max_len = 0
    for i in range(len(endpoints)):
        for j in range(i+1, min(i+10, len(endpoints))):  # 연산량 조절
            try:
                path = nx.shortest_path(sub_G, source=endpoints[i], target=endpoints[j], weight='weight')
                if len(path) > max_len:
                    max_len = len(path)
                    longest_path = path
            except nx.NetworkXNoPath:
                continue
                
    if not longest_path:
        longest_path = list(sub_G.nodes())

    # 4. 2D 좌표 리스트화 및 B-Spline 스무딩
    path_x = [points[n][0] for n in longest_path]
    path_y = [points[n][1] for n in longest_path]
    
    # 중복점 제거나 너무 촘촘한 점 줄이기 (샘플링)
    step = max(1, len(path_x) // 50)
    path_x = path_x[::step]
    path_y = path_y[::step]
    
    try:
        tck, u = splprep([path_x, path_y], s=5.0)
        u_new = np.linspace(0, 1, 100) # 100개의 매끄러운 점
        smooth_x, smooth_y = splev(u_new, tck)
    except Exception:
        smooth_x, smooth_y = path_x, path_y

    # 5. 3D 좌표 변환 (Center를 0,0,0으로 맞춤)
    centerline_3d = []
    for sx, sy in zip(smooth_x, smooth_y):
        ix, iy = int(np.clip(sx, 0, w-1)), int(np.clip(sy, 0, h-1))
        
        vx = sx - w / 2
        vy = -(sy - h / 2) # Y축 반전
        vz = depth_map[iy, ix] * (w * depth_scale)
        centerline_3d.append([vx, vy, vz])
        
    return np.array(centerline_3d)


def create_tube_mesh(path_3d: np.ndarray, radius: float, color: list, sides: int = 12) -> trimesh.Trimesh:
    """
    주어진 3D 경로(path_3d)를 따라 원통(튜브) 메쉬를 생성(Sweep)합니다.
    """
    try:
        from shapely.geometry import Point
        # 2D 원 다각형 생성
        circle_poly = Point(0, 0).buffer(radius, resolution=sides//4)
        
        # 스위핑 메쉬 생성
        mesh = trimesh.creation.sweep_polygon(circle_poly, path_3d)
        mesh.visual.vertex_colors = color
        return mesh
    except Exception as e:
        print(f"[Tube Creation Error] {e}")
        return trimesh.Trimesh()


class CableDigitalTwinBuilder:
    """
    단일 클래스에서 스켈레톤 추출, UV 기반 변형, 외피 및 혈류선 튜브 생성을 모두 관장합니다.
    """
    def __init__(self, depth_path: str, mask_path: str, tcl_path: str, depth_scale: float = 0.2,
                 connector_types: dict = None):
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.tcl_path = tcl_path
        self.depth_scale = depth_scale
        # 예: {"J14": "MS3475 W14-18P", "J26": "MS3475 W14-18P"}. 도면에서 확인한
        # 실제 커넥터 규격을 넘기면 핀 배치가 균등 원형 근사 대신 실측 배치를 쓴다.
        self.connector_types = connector_types or {}

    def build(self) -> trimesh.Trimesh:
        # 1. 이미지 로드
        depth_img = Image.open(self.depth_path).convert('L')
        mask_img = Image.open(self.mask_path).convert('L')

        target_size = (512, 512)
        depth_map = np.array(depth_img.resize(target_size, Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
        mask_map = np.array(mask_img.resize(target_size, Image.Resampling.NEAREST), dtype=np.float32) / 255.0

        # 2. 3D 스켈레톤 척추(Centerline) 추출
        centerline_3d = extract_3d_centerline(mask_map, depth_map, self.depth_scale)

        # 3. 케이블 외피(Ghost Shell) 생성
        # 얇고 부드러운 반투명 푸른색 파이프
        shell_radius = max(5.0, target_size[0] * 0.05) # 화면 크기의 5% 두께
        shell_color = [100, 149, 237, 120]  # CornflowerBlue 반투명
        shell_mesh = create_tube_mesh(centerline_3d, shell_radius, shell_color, sides=16)

        # 4. 배선(Wires) 생성 - TCL의 실제 o@(Open 검사=점대점 배선) 데이터를
        # 커넥터 UV 레이아웃에 따라 그대로 튜브화한다. 예전에는 여기서 임의의
        # 23가닥 나선형을 그렸지만, 그건 실제 배선과 무관한 장식이었다.
        from wire_builder import CircuitWireBuilder as TclCircuitWireBuilder
        bounds_min = centerline_3d.min(axis=0) - shell_radius
        bounds_max = centerline_3d.max(axis=0) + shell_radius
        wire_builder = TclCircuitWireBuilder(
            tcl_path=self.tcl_path,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            shell_radius=shell_radius,
            connector_types=self.connector_types,
        )
        wires_mesh = wire_builder.build(centerline_3d)

        # 5. 합체 및 리턴
        combined = [shell_mesh]
        if len(wires_mesh.vertices) > 0:
            combined.append(wires_mesh)
        return trimesh.util.concatenate(combined)


if __name__ == '__main__':
    # Test Script
    out_dir = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"
    
    print("[GeometryEngine] Building Skeleton-based Digital Twin...")
    builder = CableDigitalTwinBuilder(
        depth_path=os.path.join(out_dir, "Cable_depth.png"),
        mask_path=os.path.join(out_dir, "Cable_mask.png"),
        tcl_path="" # Test
    )
    twin_mesh = builder.build()
    
    out_glb = os.path.join(out_dir, "Cable_DigitalTwin_V2.glb")
    twin_mesh.export(out_glb)
    print(f"-> Successfully exported new skeleton hybrid 3D model to {out_glb}")


