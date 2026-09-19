import cv2
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from detectors import DrawingShapeDetector, CableAnalyzer, Visualizer

class CableImageViewer:
    def __init__(self, root, image_dir):
        self.root = root
        self.root.title("Drawing Shape Analyzer")
        self.root.geometry("1300x800") # 넓이를 키워 우측 패널 공간 확보
        
        self.image_dir = image_dir
        self.image_paths = []
        self.current_idx = 0
        
        self.analyzer = CableAnalyzer()
        # 도면 전용 검출기 장착
        self.analyzer.add_detector('drawing_detector', DrawingShapeDetector(min_area=50, epsilon_factor=0.01))
        
        self.load_image_list()
        
        # ---------------- UI 구성 ----------------
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.info_label = tk.Label(self.top_frame, text="Loading...", font=("Arial", 12, "bold"))
        self.info_label.pack(side=tk.TOP)
        
        self.view_mode = tk.StringVar(value="final")
        modes = [
            ("1. 원본 이미지", "original"), 
            ("2. 도면 이진화(Binary)", "step2"), 
            ("3. 윤곽선(Contours)", "step3"), 
            ("4. 최종 분석(Final)", "final")
        ]
                 
        self.mode_frame = tk.Frame(self.top_frame)
        self.mode_frame.pack(side=tk.TOP, pady=5)
        for text, mode in modes:
            rb = ttk.Radiobutton(self.mode_frame, text=text, variable=self.view_mode, 
                                 value=mode, command=self.display_current_image)
            rb.pack(side=tk.LEFT, padx=10)
            
        # 메인 영역 분할 (좌측 캔버스, 우측 요약 패널)
        self.main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.main_pane, bg="black")
        self.main_pane.add(self.canvas, minsize=800)
        
        self.summary_frame = tk.Frame(self.main_pane, width=250, bg="#2b2b2b")
        self.main_pane.add(self.summary_frame, minsize=200)
        
        # 요약 텍스트 라벨
        self.summary_title = tk.Label(self.summary_frame, text="도형 검출 요약 (Summary)", 
                                      font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white")
        self.summary_title.pack(pady=10)
        
        self.summary_text = tk.Text(self.summary_frame, bg="#2b2b2b", fg="white", 
                                    font=("Consolas", 12), borderwidth=0, state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10)
        
        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        self.btn_prev = ttk.Button(self.bottom_frame, text="이전 (A / Left)", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=50)
        
        self.btn_next = ttk.Button(self.bottom_frame, text="다음 (D / Right)", command=self.next_image)
        self.btn_next.pack(side=tk.RIGHT, padx=50)
        
        # ---------------- 단축키 ----------------
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.next_image())
        self.root.bind("<Return>", lambda e: self.next_image())
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.bind("<q>", lambda e: self.root.quit())
        
        self.root.bind("1", lambda e: [self.view_mode.set("original"), self.display_current_image()])
        self.root.bind("2", lambda e: [self.view_mode.set("step2"), self.display_current_image()])
        self.root.bind("3", lambda e: [self.view_mode.set("step3"), self.display_current_image()])
        self.root.bind("4", lambda e: [self.view_mode.set("final"), self.display_current_image()])
        
        self.display_current_image()
        
    def load_image_list(self):
        valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        for ext in valid_exts:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext.upper())))
        
        self.image_paths = sorted(list(set(self.image_paths)))
        if not self.image_paths:
            print(f"[오류] 다음 경로에서 이미지를 찾을 수 없습니다:\n{self.image_dir}")
            
    def load_image_cv2(self, filepath):
        img_array = np.fromfile(filepath, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def update_summary(self, shapes):
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        total = 0
        summary_str = ""
        
        # 순서대로 렌더링하기 위한 매핑
        order = [
            ('triangle', '🔺 삼각형 (3)'),
            ('rectangle', '🟩 사각형 (4)'),
            ('pentagon', '💠 오각형 (5)'),
            ('hexagon', '🛑 육각형 (6)'),
            ('arrow_heptagon', '▶️ 화살표 등 (7)'),
            ('circle_complex', '🔵 원형 (8+)'),
            ('line_like', '📏 긴 선(Line)'),
            ('text_like', '🔠 텍스트/기호(걸러짐)')
        ]
        
        for key, name in order:
            count = len(shapes.get(key, []))
            total += count
            summary_str += f"\n {name}\n   -> {count} 개\n"
            
        summary_str += f"\n\n 🏆 총 검출 수: {total} 개"
        
        self.summary_text.insert(tk.END, summary_str)
        self.summary_text.config(state=tk.DISABLED)
        
    def display_current_image(self):
        if not self.image_paths:
            return
            
        img_path = self.image_paths[self.current_idx]
        original_img = self.load_image_cv2(img_path)
        
        filename = os.path.basename(img_path)
        self.info_label.config(text=f"[{self.current_idx + 1} / {len(self.image_paths)}] {filename}")
        
        if original_img is None:
            img_to_show = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(img_to_show, "Failed to load image", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.update_summary({})
        else:
            # 1. 일단 이미지 전체 분석 수행
            results = self.analyzer.analyze(original_img)
            
            # 우측 패널 요약 업데이트
            shape_data = results.get('drawing_detector', {}).get('shapes', {})
            self.update_summary(shape_data)
            
            # 2. 현재 선택된 라디오 버튼 모드에 따라 표시할 이미지 결정
            mode = self.view_mode.get()
            shape_debug = results.get('drawing_detector', {}).get('debug', {})
            
            if mode == "original":
                img_to_show = original_img.copy()
            elif mode == "step2" and 'step2_binary' in shape_debug:
                img_to_show = shape_debug['step2_binary']
            elif mode == "step3" and 'step3_contours' in shape_debug:
                img_to_show = shape_debug['step3_contours']
            elif mode == "final":
                img_to_show = Visualizer.draw_results(original_img, results)
            else:
                img_to_show = original_img.copy()
                
            if len(img_to_show.shape) == 2:
                img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_GRAY2RGB)
            else:
                img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
                
        pil_img = Image.fromarray(img_to_show)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 800, 650
            
        img_width, img_height = pil_img.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (int(img_width*ratio), int(img_height*ratio))
        
        if new_size[0] > 0 and new_size[1] > 0:
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
        
    def prev_image(self):
        if self.image_paths:
            self.current_idx = (self.current_idx - 1) % len(self.image_paths)
            self.display_current_image()
            
    def next_image(self):
        if self.image_paths:
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
            self.display_current_image()

def main():
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PDF_File", "Image")
    root = tk.Tk()
    app = CableImageViewer(root, image_dir)
    root.bind("<Configure>", lambda e: app.display_current_image() if e.widget == root else None)
    root.mainloop()

if __name__ == "__main__":
    main()
