import cv2
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class VectorSketchViewer:
    def __init__(self, root, image_dir):
        self.root = root
        self.root.title("OpenCV Vector Sketch (끊김 방지 복원 과정)")
        self.root.geometry("1100x800")
        
        self.image_dir = image_dir
        self.image_paths = []
        self.current_idx = 0
        
        self.load_image_list()
        
        # ---------------- UI 구성 ----------------
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.info_label = tk.Label(self.top_frame, text="Loading...", font=("Arial", 12, "bold"))
        self.info_label.pack(side=tk.TOP)
        
        self.view_mode = tk.StringVar(value="step4_sketch")
        modes = [
            ("1. 원본 이미지", "step1_original"), 
            ("2. 이진화 (선 끊김 발생 가능)", "step2_thresh"),
            ("3. 모폴로지 팽창 (선 이어붙이기)", "step3_dilate"),
            ("4. 최종 벡터 스케치", "step4_sketch")
        ]
                 
        self.mode_frame = tk.Frame(self.top_frame)
        self.mode_frame.pack(side=tk.TOP, pady=5)
        for text, mode in modes:
            rb = ttk.Radiobutton(self.mode_frame, text=text, variable=self.view_mode, 
                                 value=mode, command=self.display_current_image)
            rb.pack(side=tk.LEFT, padx=10)
            
        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
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
        
        self.root.bind("1", lambda e: [self.view_mode.set("step1_original"), self.display_current_image()])
        self.root.bind("2", lambda e: [self.view_mode.set("step2_thresh"), self.display_current_image()])
        self.root.bind("3", lambda e: [self.view_mode.set("step3_dilate"), self.display_current_image()])
        self.root.bind("4", lambda e: [self.view_mode.set("step4_sketch"), self.display_current_image()])
        
        self.display_current_image()
        
    def load_image_list(self):
        valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        for ext in valid_exts:
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(self.image_dir, ext.upper())))
        self.image_paths = sorted(list(set(self.image_paths)))
            
    def load_image_cv2(self, filepath):
        img_array = np.fromfile(filepath, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def process_image(self, image):
        # 1. 흑백 및 이진화 (선이 끊어질 수 있는 상태)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 2. 모폴로지 팽창 (끊어진 선을 강제로 이어붙임)
        # 2x2 픽셀 크기의 붓으로 하얀색 선을 뚱뚱하게 덧칠합니다.
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # 3. 끊김이 복구된 이미지에서 윤곽선 100% 추출
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 4. 새하얀 백지에 벡터 좌표만으로 스케치
        canvas = np.ones_like(image) * 255
        cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)
        
        cv2.putText(canvas, f"Total Vector Contours: {len(contours)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    
        return thresh, dilated, canvas
        
    def display_current_image(self):
        if not self.image_paths:
            return
            
        img_path = self.image_paths[self.current_idx]
        original_img = self.load_image_cv2(img_path)
        
        filename = os.path.basename(img_path)
        self.info_label.config(text=f"[{self.current_idx + 1} / {len(self.image_paths)}] {filename}")
        
        thresh_img, dilated_img, sketch_img = self.process_image(original_img)
        
        mode = self.view_mode.get()
        if mode == "step1_original":
            img_to_show = original_img.copy()
        elif mode == "step2_thresh":
            # 이진화 이미지는 흑백(1채널)이므로 RGB로 변환해서 출력
            img_to_show = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        elif mode == "step3_dilate":
            img_to_show = cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR)
        else:
            img_to_show = sketch_img
            
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 1000, 700
            
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

if __name__ == "__main__":
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PDF_File", "Image")
    root = tk.Tk()
    app = VectorSketchViewer(root, image_dir)
    root.bind("<Configure>", lambda e: app.display_current_image() if e.widget == root else None)
    root.mainloop()
