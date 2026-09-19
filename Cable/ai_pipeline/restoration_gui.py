import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import pytesseract

# Tesseract 경로 설정 (사용자 환경에 맞게 수정)
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

class ImageRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("도면 복원 & OCR 위치 텍스트 추출 툴")
        self.root.geometry("1200x800")
        
        # 상태 변수
        self.filepath = None
        self.original_img = None
        self.processed_img = None
        self.ocr_result_img = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # --- 좌측 컨트롤 패널 ---
        control_frame = tk.Frame(self.root, width=250, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Button(control_frame, text="📁 파일 불러오기 (이미지/PDF)", command=self.load_file, height=2).pack(fill=tk.X, pady=(0, 15))
        
        # 노이즈 제거 (h)
        tk.Label(control_frame, text="노이즈 제거 강도 (h)").pack(anchor='w')
        self.h_slider = tk.Scale(control_frame, from_=0, to=30, orient=tk.HORIZONTAL)
        self.h_slider.set(10)
        self.h_slider.pack(fill=tk.X, pady=(0, 10))
        
        # 적응형 이진화 Block Size
        tk.Label(control_frame, text="이진화 블록 크기 (홀수)").pack(anchor='w')
        self.block_slider = tk.Scale(control_frame, from_=3, to=99, resolution=2, orient=tk.HORIZONTAL)
        self.block_slider.set(21)
        self.block_slider.pack(fill=tk.X, pady=(0, 10))
        
        # 적응형 이진화 C
        tk.Label(control_frame, text="이진화 상수 (C)").pack(anchor='w')
        self.c_slider = tk.Scale(control_frame, from_=0, to=30, orient=tk.HORIZONTAL)
        self.c_slider.set(10)
        self.c_slider.pack(fill=tk.X, pady=(0, 10))
        
        # 모폴로지 (팽창)
        tk.Label(control_frame, text="모폴로지 팽창 (선 굵게)").pack(anchor='w')
        self.morph_slider = tk.Scale(control_frame, from_=0, to=5, orient=tk.HORIZONTAL)
        self.morph_slider.set(1)
        self.morph_slider.pack(fill=tk.X, pady=(0, 20))
        
        tk.Button(control_frame, text="▶ 1. 복원 필터 적용", command=self.process_image, bg="lightblue", height=2).pack(fill=tk.X, pady=(0, 10))
        
        # OCR 버튼 추가
        tk.Button(control_frame, text="🔍 2. 도면 OCR 텍스트 추출 (위치 포함)", command=self.extract_ocr_text, bg="lightgreen", height=2).pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(control_frame, text="💾 결과 이미지 저장", command=self.save_image, height=2).pack(fill=tk.X)
        
        # --- 우측 메인 패널 (위: 이미지, 아래: 텍스트 추출 결과) ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 1. 상단 이미지 뷰어
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        orig_frame = tk.Frame(image_frame)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(orig_frame, text="[ 원본 이미지 ]").pack()
        self.lbl_orig = tk.Label(orig_frame, bg="gray")
        self.lbl_orig.pack(fill=tk.BOTH, expand=True)
        
        proc_frame = tk.Frame(image_frame)
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        tk.Label(proc_frame, text="[ 필터 / OCR 결과 이미지 ]").pack()
        self.lbl_proc = tk.Label(proc_frame, bg="gray")
        self.lbl_proc.pack(fill=tk.BOTH, expand=True)
        
        # 2. 하단 텍스트 추출 뷰어
        text_frame = tk.Frame(main_frame, height=250)
        text_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        text_frame.pack_propagate(False) # 높이 고정
        
        tk.Label(text_frame, text="[ 추출된 텍스트 목록 (X, Y, W, H 포함) ]").pack(anchor='w')
        self.txt_extract = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=12)
        self.txt_extract.pack(fill=tk.BOTH, expand=True)

    def load_file(self):
        self.filepath = filedialog.askopenfilename(
            filetypes=[("All Supported", "*.pdf;*.png;*.jpg;*.jpeg;*.bmp"), 
                       ("PDF Files", "*.pdf"), 
                       ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not self.filepath:
            return
            
        self.txt_extract.delete('1.0', tk.END) # 텍스트 초기화
        
        if self.filepath.lower().endswith('.pdf'):
            self.load_pdf(self.filepath)
        else:
            self.original_img = cv2.imread(self.filepath)
            self.txt_extract.insert(tk.END, "이미지 파일이 로드되었습니다. '도면 OCR 추출' 버튼을 눌러주세요.\n")
            
        self.display_image(self.original_img, self.lbl_orig)
        self.lbl_proc.configure(image='')
        self.processed_img = None
        self.ocr_result_img = None

    def load_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            page = doc[0] # 첫 페이지 기준
            
            # PDF 고해상도 렌더링 (줌인 2.5배)
            zoom = 2.5 
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # PyMuPDF Pixmap -> OpenCV Numpy 배열 변환
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            if pix.n == 4: # RGBA
                self.original_img = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3: # RGB
                self.original_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            else: # GRAY
                self.original_img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                
            self.txt_extract.insert(tk.END, f"PDF 첫 페이지가 성공적으로 렌더링 되었습니다.\n'도면 OCR 추출' 버튼을 눌러 위치와 텍스트를 추출하세요.\n\n")
            doc.close()
            
        except Exception as e:
            messagebox.showerror("오류", f"PDF를 불러오는 중 오류가 발생했습니다:\n{e}")

    def process_image(self):
        if self.original_img is None:
            messagebox.showwarning("경고", "먼저 파일을 불러와주세요.")
            return

        h_val = self.h_slider.get()
        block_size = self.block_slider.get()
        c_val = self.c_slider.get()
        morph_iter = self.morph_slider.get()

        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=h_val, templateWindowSize=7, searchWindowSize=21)

        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)

        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val)

        if morph_iter > 0:
            inverted = cv2.bitwise_not(binary)
            kernel_morph = np.ones((2,2), np.uint8)
            dilated = cv2.dilate(inverted, kernel_morph, iterations=morph_iter)
            self.processed_img = cv2.bitwise_not(dilated)
        else:
            self.processed_img = binary

        self.display_image(self.processed_img, self.lbl_proc)
        
    def extract_ocr_text(self):
        # 복원된 이미지가 없으면 원본으로 수행
        target_img = self.processed_img if self.processed_img is not None else self.original_img
        if target_img is None:
            messagebox.showwarning("경고", "먼저 파일을 불러와주세요.")
            return

        # OpenCV 이미지는 컬러로 변환해서 바운딩 박스를 그릴 준비
        if len(target_img.shape) == 2:
            display_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = target_img.copy()

        self.txt_extract.insert(tk.END, "--- Tesseract OCR 위치 기반 텍스트 추출 시작 ---\n")
        self.txt_extract.update()

        try:
            # Tesseract OCR 실행 (dict 형태로 좌상단 좌표 및 w, h 포함 반환)
            # psm 11: Sparse text with as much text as possible in no particular order.
            data = pytesseract.image_to_data(target_img, lang="eng", config="--psm 11", output_type=pytesseract.Output.DICT)
            
            n = len(data['text'])
            count = 0
            
            for i in range(n):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != "-1" else -1
                
                # 비어있거나 신뢰도가 낮은 텍스트는 건너뜀
                if not text or conf < 0:
                    continue
                
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # 이미지 위에 빨간색 박스와 텍스트 그리기
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 텍스트 박스에 정보 추가
                log_text = f"X: {x:4d} | Y: {y:4d} | W: {w:3d} | H: {h:3d} | Text: {text}\n"
                self.txt_extract.insert(tk.END, log_text)
                count += 1
                
            self.txt_extract.insert(tk.END, f"총 {count}개의 텍스트를 같은 위치에서 성공적으로 추출했습니다.\n\n")
            self.txt_extract.see(tk.END)
            
            self.ocr_result_img = display_img
            self.display_image(self.ocr_result_img, self.lbl_proc)
            
        except Exception as e:
            messagebox.showerror("OCR 오류", f"Tesseract OCR 실행 중 오류가 발생했습니다.\nTesseract가 정상 설치되어 있는지 확인하세요.\n{e}")

    def display_image(self, cv_img, label_widget):
        h, w = cv_img.shape[:2]
        max_size = 500
        scale = min(max_size/w, max_size/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label_widget.configure(image=img_tk)
        label_widget.image = img_tk

    def save_image(self):
        target = self.ocr_result_img if self.ocr_result_img is not None else self.processed_img
        if target is None:
            messagebox.showwarning("경고", "저장할 복원/OCR 결과 이미지가 없습니다.")
            return
            
        save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, target)
            messagebox.showinfo("완료", "성공적으로 저장되었습니다.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRestorationGUI(root)
    root.mainloop()
