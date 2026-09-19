import cv2
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from detectors import DrawingShapeDetector, CableAnalyzer, Visualizer, find_optimal_kernel
from pipeline.runner import run_pipeline, summarize
from pipeline.visualize import LAYERS, render_view, bgr_to_hex

class CableMasterDashboard:
    def __init__(self, root, image_dir):
        self.root = root
        self.root.title("Cable Master Dashboard - 통합 분석 시스템")
        self.root.geometry("1300x850")
        
        self.image_dir = image_dir
        self.image_paths = []
        self.current_idx = 0
        
        # 도형 검출기 초기화
        self.analyzer = CableAnalyzer()
        self.analyzer.add_detector('drawing_detector', DrawingShapeDetector(min_area=50, epsilon_factor=0.01))
        
        self.load_image_list()
        
        # ==========================================
        # 1. 상단 컨트롤 패널 (공통)
        # ==========================================
        self.top_frame = tk.Frame(root, bg="#1e1e1e", pady=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.btn_prev = tk.Button(self.top_frame, text="◀ 이전 도면 (A)", command=self.prev_image, 
                                  font=("Arial", 11, "bold"), bg="#3a3a3a", fg="white", relief=tk.FLAT, padx=15)
        self.btn_prev.pack(side=tk.LEFT, padx=20)
        
        self.info_label = tk.Label(self.top_frame, text="Loading...", font=("Consolas", 14, "bold"), bg="#1e1e1e", fg="#00ff00")
        self.info_label.pack(side=tk.LEFT, expand=True)
        
        self.btn_next = tk.Button(self.top_frame, text="다음 도면 (D) ▶", command=self.next_image, 
                                  font=("Arial", 11, "bold"), bg="#3a3a3a", fg="white", relief=tk.FLAT, padx=15)
        self.btn_next.pack(side=tk.RIGHT, padx=20)
        
        # ==========================================
        # 2. 메인 탭(Notebook) 구성
        # ==========================================
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("Arial", 12, "bold"), padding=[10, 5])
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 탭 변경 이벤트 바인딩
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.update_dashboard())
        
        # --- [탭 1] 도형 분석 모드 ---
        self.tab_shape = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_shape, text="🧩 도형 분석 모드 (Shape Analyzer)")
        self.setup_shape_tab()
        
        # --- [탭 2] 벡터 스케치 모드 ---
        self.tab_vector = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_vector, text="✍️ 벡터 스케치 모드 (Vector Restoration)")
        self.setup_vector_tab()

        # --- [탭 3] 단계별 파이프라인 (pipeline/ 모듈) ---
        self.tab_pipeline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pipeline, text="🧪 단계별 파이프라인 (Pipeline)")
        self.setup_pipeline_tab()

        # ==========================================
        # 단축키 바인딩
        # ==========================================
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.next_image())
        
        # 초기 화면 로드
        self.update_dashboard()
        
    def setup_shape_tab(self):
        # 탭1 전용 라디오 버튼
        self.shape_mode = tk.StringVar(value="final")
        modes = [
            ("1. 원본", "original"), 
            ("2. 이진화", "step2"), 
            ("3. 윤곽선", "step3"), 
            ("4. 최종 분석(Final)", "final")
        ]
        
        mode_frame = tk.Frame(self.tab_shape)
        mode_frame.pack(side=tk.TOP, pady=5)
        for text, mode in modes:
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.shape_mode, 
                                 value=mode, command=self.update_dashboard)
            rb.pack(side=tk.LEFT, padx=15)
            
        # 메인 영역 (캔버스 + 우측 패널)
        pane = tk.PanedWindow(self.tab_shape, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.shape_canvas = tk.Canvas(pane, bg="black")
        pane.add(self.shape_canvas, minsize=800)
        
        summary_frame = tk.Frame(pane, width=250, bg="#2b2b2b")
        pane.add(summary_frame, minsize=250)
        
        tk.Label(summary_frame, text="도형 검출 통계", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)
        self.summary_text = tk.Text(summary_frame, bg="#2b2b2b", fg="#00ff00", font=("Consolas", 12), borderwidth=0, state=tk.DISABLED, height=15)
        self.summary_text.pack(fill=tk.X, padx=10, pady=5)
        
        # 레이어 토글(Checkboxes)
        self.show_flags = {
            'polylines': tk.BooleanVar(value=True),
            'arrow_lines': tk.BooleanVar(value=True), # [NEW] 지시선 토글
            'bubbles': tk.BooleanVar(value=True),
            'rects': tk.BooleanVar(value=True),
            'arrows': tk.BooleanVar(value=True),
            'text_blocks': tk.BooleanVar(value=True)
        }
        
        toggle_frame = tk.Frame(summary_frame, bg="#2b2b2b")
        toggle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        tk.Label(toggle_frame, text="[ 레이어 켜기/끄기 ]", font=("Arial", 12, "bold"), bg="#2b2b2b", fg="yellow").pack(pady=5)
        
        tk.Checkbutton(toggle_frame, text="폴리라인 배선 (핫핑크)", variable=self.show_flags['polylines'], bg="#2b2b2b", fg="white", selectcolor="#444444", command=self.update_dashboard).pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="화살표 지시선 (노란색)", variable=self.show_flags['arrow_lines'], bg="#2b2b2b", fg="white", selectcolor="#444444", command=self.update_dashboard).pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="부품 번호 (주황색)", variable=self.show_flags['bubbles'], bg="#2b2b2b", fg="white", selectcolor="#444444", command=self.update_dashboard).pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="핀 박스 (초록색)", variable=self.show_flags['rects'], bg="#2b2b2b", fg="white", selectcolor="#444444", command=self.update_dashboard).pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="화살표 촉 (빨간색)", variable=self.show_flags['arrows'], bg="#2b2b2b", fg="white", selectcolor="#444444", command=self.update_dashboard).pack(anchor=tk.W)
        tk.Checkbutton(toggle_frame, text="기본 텍스트 (흰색)", variable=self.show_flags['text_blocks'], bg="#2b2b2b", fg="white", selectcolor="#444444", command=self.update_dashboard).pack(anchor=tk.W)
        
    def setup_vector_tab(self):
        # 탭2 전용 라디오 버튼
        self.vector_mode = tk.StringVar(value="step4_sketch")
        modes = [
            ("1. 원본", "step1_original"), 
            ("2. 이진화 (선 끊김 관찰)", "step2_thresh"),
            ("3. 팽창 (선 이어붙이기)", "step3_dilate"),
            ("4. 최종 100% 벡터 스케치", "step4_sketch")
        ]
        
        vector_control_frame = tk.Frame(self.tab_vector)
        vector_control_frame.pack(side=tk.TOP, pady=5)
        for text, mode in modes:
            rb = ttk.Radiobutton(vector_control_frame, text=text, variable=self.vector_mode, 
                                 value=mode, command=self.update_dashboard)
            rb.pack(side=tk.LEFT, padx=5)
            
        ttk.Separator(vector_control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=15, fill=tk.Y)
        
        self.kernel_mode = tk.StringVar(value="fixed_1x1")
        ttk.Label(vector_control_frame, text="[붓 크기 설정]:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(vector_control_frame, text="점선/원본 보존 (1x1)", variable=self.kernel_mode, value="fixed_1x1", command=self.update_dashboard).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(vector_control_frame, text="단순 닫기 (2x2)", variable=self.kernel_mode, value="fixed_2x2", command=self.update_dashboard).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(vector_control_frame, text="엘보우 탐색 (자동)", variable=self.kernel_mode, value="auto_elbow", command=self.update_dashboard).pack(side=tk.LEFT, padx=5)
        
        self.vector_canvas = tk.Canvas(self.tab_vector, bg="gray")
        self.vector_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def setup_pipeline_tab(self):
        # 탭3 전용 라디오 버튼
        self.pipeline_mode = tk.StringVar(value="overlay")
        modes = [
            ("0. 원본", "original"),
            ("1. 이진화", "binary"),
            ("2. 그래픽 제거 후 잔여", "residual"),
            ("3. 글자만", "text_only"),
            ("4. 레이어 뷰", "overlay")
        ]

        mode_frame = tk.Frame(self.tab_pipeline)
        mode_frame.pack(side=tk.TOP, pady=5)
        for text, mode in modes:
            rb = ttk.Radiobutton(mode_frame, text=text, variable=self.pipeline_mode,
                                 value=mode, command=self.update_dashboard)
            rb.pack(side=tk.LEFT, padx=15)

        # 메인 영역 (캔버스 + 우측 패널)
        pane = tk.PanedWindow(self.tab_pipeline, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, pady=5)

        self.pipeline_canvas = tk.Canvas(pane, bg="black")
        pane.add(self.pipeline_canvas, minsize=800, stretch="always")

        side = tk.Frame(pane, width=270, bg="#2b2b2b")
        pane.add(side, minsize=270, width=270, stretch="never")

        tk.Label(side, text="단계별 검출 통계", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)
        self.pipeline_summary = tk.Text(side, bg="#2b2b2b", fg="#00ff00", font=("Consolas", 11), borderwidth=0, state=tk.DISABLED, height=19)
        self.pipeline_summary.pack(fill=tk.X, padx=10, pady=5)

        # 레이어 토글 (4. 레이어 뷰에서 적용)
        tk.Label(side, text="[ 레이어 켜기/끄기 ]", font=("Arial", 12, "bold"), bg="#2b2b2b", fg="yellow").pack(pady=5)
        btn_frame = tk.Frame(side, bg="#2b2b2b")
        btn_frame.pack()
        tk.Button(btn_frame, text="모두 켜기", command=lambda: self.set_pipeline_layers(True),
                  bg="#3a3a3a", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="모두 끄기", command=lambda: self.set_pipeline_layers(False),
                  bg="#3a3a3a", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=5)

        self.pipeline_flags = {}
        toggle_frame = tk.Frame(side, bg="#2b2b2b")
        toggle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        for key, label, color, _ in LAYERS:
            var = tk.BooleanVar(value=True)
            self.pipeline_flags[key] = var
            row = tk.Frame(toggle_frame, bg="#2b2b2b")
            row.pack(anchor=tk.W)
            tk.Label(row, text="■", fg=bgr_to_hex(color), bg="#2b2b2b", font=("Arial", 13)).pack(side=tk.LEFT)
            tk.Checkbutton(row, text=label, variable=var, bg="#2b2b2b", fg="white", selectcolor="#444444",
                           activebackground="#2b2b2b", command=self.update_dashboard).pack(side=tk.LEFT)

        # 파이프라인 결과 캐시 (창 크기 변경/레이어 토글 때 재계산하지 않도록 현재 이미지 1장만 보관)
        self.pipeline_cache = (None, None)

    def set_pipeline_layers(self, on):
        for var in self.pipeline_flags.values():
            var.set(on)
        self.update_dashboard()

    def render_pipeline_tab(self, original_img, img_path):
        cached_path, res = self.pipeline_cache
        if cached_path != img_path:
            res = run_pipeline(original_img)
            self.pipeline_cache = (img_path, res)

        s = summarize(res)
        summary_str = (
            f"글자 높이(char_h): {s['char_h']}px\n"
            f"선 두께(stroke):   {s['stroke_w']}px\n"
            + "=" * 24 + "\n"
            f"[그래픽 - 지운 것]\n"
            f" 원           : {s['circle']}\n"
            f" 가로선       : {s['h_line']}\n"
            f" 세로선       : {s['v_line']}\n"
            f" 점선         : {s['dash_line']}\n"
            f" 사선         : {s['diag_line']}\n"
            f" 네모/삼각/다각: {s['rect']}/{s['triangle']}/{s['polygon']}\n"
            f"[잔여 조각]\n"
            f" 접속점       : {s['dot']}\n"
            f" 화살촉       : {s['arrowhead']}\n"
            f" 그래픽 잔재  : {s['remnant']}\n"
            f" 떨어진 대시  : {s['dash']}\n"
            + "=" * 24 + "\n"
            f" 텍스트 박스  : {s['text']}\n"
        )
        self.pipeline_summary.config(state=tk.NORMAL)
        self.pipeline_summary.delete(1.0, tk.END)
        self.pipeline_summary.insert(tk.END, summary_str)
        self.pipeline_summary.config(state=tk.DISABLED)

        enabled = {k for k, var in self.pipeline_flags.items() if var.get()}
        img_to_show = render_view(res, self.pipeline_mode.get(), enabled)
        self.draw_on_canvas(self.pipeline_canvas, img_to_show)

    def load_image_list(self):
        valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        for ext in valid_exts:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext.upper())))
        self.image_paths = sorted(list(set(self.image_paths)))
        
    def prev_image(self):
        if self.image_paths:
            self.current_idx = (self.current_idx - 1) % len(self.image_paths)
            self.update_dashboard()
            
    def next_image(self):
        if self.image_paths:
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
            self.update_dashboard()

    def update_dashboard(self):
        if not self.image_paths:
            self.info_label.config(text="이미지를 찾을 수 없습니다.")
            return
            
        img_path = self.image_paths[self.current_idx]
        original_img = np.fromfile(img_path, np.uint8)
        original_img = cv2.imdecode(original_img, cv2.IMREAD_COLOR)
        
        filename = os.path.basename(img_path)
        self.info_label.config(text=f"[{self.current_idx + 1} / {len(self.image_paths)}]  {filename}")
        
        # 현재 열려있는 탭 확인
        current_tab_idx = self.notebook.index(self.notebook.select())
        
        if current_tab_idx == 0:
            self.render_shape_tab(original_img)
        elif current_tab_idx == 1:
            self.render_vector_tab(original_img)
        elif current_tab_idx == 2:
            self.render_pipeline_tab(original_img, img_path)
            
    def draw_on_canvas(self, canvas_widget, cv2_img):
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        canvas_width = canvas_widget.winfo_width()
        canvas_height = canvas_widget.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 1000, 700
            
        img_width, img_height = pil_img.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (int(img_width*ratio), int(img_height*ratio))
        
        if new_size[0] > 0 and new_size[1] > 0:
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            
        photo = ImageTk.PhotoImage(pil_img)
        canvas_widget.delete("all")
        canvas_widget.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        canvas_widget.image = photo # 가비지 컬렉션 방지
        
    def render_shape_tab(self, original_img):
        results = self.analyzer.analyze(original_img)
        shape_data = results.get('drawing_detector', {}).get('shapes', {})
        shape_debug = results.get('drawing_detector', {}).get('debug', {})
        
        # 통계 패널 업데이트
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        total = 0
        summary_str = ""
        
        # Auto-Tuning 결과 피드백 추가
        auto_thresh = shape_debug.get('auto_thresh_val', 200)
        auto_kernel = shape_debug.get('auto_kernel_size', 1)
        summary_str += f"🤖 Auto-Tuning 완료\n"
        summary_str += f"- 최적 임계값: {int(auto_thresh)}\n"
        summary_str += f"- 팽창 붓굵기: {auto_kernel}x{auto_kernel}\n"
        summary_str += "="*20 + "\n"
        
        order = [
            ('polylines', '[핫핑크] 메인 결선 라인 (경로)'), 
            ('arrow_lines', '[노란색] 화살표 지시선 (사선)'),
            ('bubbles', '[주황색] 부품 번호 기호 (원형)'),
            ('rects', '[초록색] 커넥터 핀 박스 (사각)'), 
            ('arrows', '[빨간색] 화살표 촉 (과녁)'),
            ('text_blocks', '[흰색] 기본 텍스트 패치')
        ]
        
        summary_str += "=======================\n   기하학 토폴로지 추출 결과\n=======================\n"
        for key, desc in order:
            cnt = len(results['drawing_detector']['shapes'].get(key, []))
            summary_str += f"{desc}: {cnt}개\n"
        self.summary_text.insert(tk.END, summary_str)
        self.summary_text.config(state=tk.DISABLED)
        
        # 이미지 모드 선택
        mode = self.shape_mode.get()
        if mode == "original": img_to_show = original_img.copy()
        elif mode == "step2" and 'step2_binary' in shape_debug:
            img_to_show = cv2.cvtColor(shape_debug['step2_binary'], cv2.COLOR_GRAY2BGR)
        elif mode == "step3" and 'step3_contours' in shape_debug:
            img_to_show = shape_debug['step3_contours']
        else:
            # 레이어 켜기/끄기 옵션 적용
            import copy
            filtered_results = copy.deepcopy(results)
            filtered_shapes = filtered_results.get('drawing_detector', {}).get('shapes', {})
            for key, flag_var in self.show_flags.items():
                if not flag_var.get():
                    filtered_shapes[key] = []
                    
            img_to_show = Visualizer.draw_results(original_img, filtered_results)
            
        self.draw_on_canvas(self.shape_canvas, img_to_show)
        
    def render_vector_tab(self, original_img):
        # 1. 워터마크 투명화 필터
        gray = np.max(original_img, axis=2).astype(np.uint8)
        
        # 2. CLAHE + Otsu (단일 이진화망 - 가장 빠르고 직관적임)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        thresh_val, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # 3. 커널 크기 결정 (UI 선택에 따라)
        if hasattr(self, 'kernel_mode'):
            mode_val = self.kernel_mode.get()
            if mode_val == "auto_elbow":
                kernel_size = find_optimal_kernel(thresh)
            elif mode_val == "fixed_2x2":
                kernel_size = 2
            else:
                kernel_size = 1  # fixed_1x1 (점선 보존)
        else:
            kernel_size = 1
            
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        else:
            processed = thresh.copy()
            
        # UI 상태에 따라 보여줄 이미지 결정
        mode = self.vector_mode.get() if hasattr(self, 'vector_mode') else "step4_sketch"
        
        if mode == "step1_original":
            img_to_show = original_img
        elif mode == "step2_thresh":
            img_to_show = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        elif mode == "step3_dilate":
            img_to_show = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            # 5. 최종 좌표 추출용 날것(Raw) 스케치 렌더링
            # 헛짓거리(다림질, 안티앨리어싱) 싹 다 버리고 데이터 훼손 없이 좌표를 뽑을 수 있는 날것 픽셀
            sketch = cv2.bitwise_not(processed)
            sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
            
            # 텍스트
            info_text = f"Raw Coordinates Ready: Thresh={int(thresh_val)}, Kernel={kernel_size}x{kernel_size}"
            cv2.putText(sketch, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(sketch, "Status: Pixel Perfect (No Filters)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            img_to_show = sketch
            
        self.draw_on_canvas(self.vector_canvas, img_to_show)

if __name__ == "__main__":
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PDF_File", "Image")
    root = tk.Tk()
    app = CableMasterDashboard(root, image_dir)
    root.bind("<Configure>", lambda e: app.update_dashboard() if e.widget == root else None)
    root.mainloop()
