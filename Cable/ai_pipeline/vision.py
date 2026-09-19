import os
import torch
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from transformers import pipeline as hf_pipeline
import rembg


class VisionModelBase(ABC):
    """
    AI Vision 추론을 위한 추상 기본 클래스
    """
    def __init__(self, device: str = None):
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load_model(self):
        pass


class RMBGExtractor(VisionModelBase):
    """
    배경 제거를 위한 모델 클래스 (rembg 라이브러리 기반 BiRefNet 사용)
    """
    def __init__(self, device: str = None):
        super().__init__(device)
        self.session = None

    def load_model(self):
        print(f"[RMBGExtractor] Loading rembg session (birefnet-general)...")
        # birefnet-general은 현존 오픈소스 Matting 중 가장 엣지가 부드럽고 정밀함
        self.session = rembg.new_session("birefnet-general")

    def extract_mask(self, image_path: str) -> tuple[Image.Image, Image.Image]:
        """단일 이미지에 대해 배경이 제거된 RGBA 이미지와 Alpha Mask 이미지를 반환합니다."""
        orig_img = Image.open(image_path).convert("RGB")
        
        # 배경 제거 수행
        rgba_img = rembg.remove(orig_img, session=self.session)
        
        # 알파 채널을 마스크로 추출
        mask_pil = rgba_img.split()[3]
        return rgba_img, mask_pil


class DepthEstimator(VisionModelBase):
    """
    Depth Anything v2 기반의 3D 깊이 맵 추출 클래스
    """
    def __init__(self, device: str = None):
        super().__init__(device)
        self.pipe = None

    def load_model(self):
        # HuggingFace pipeline을 사용해 Depth Anything v2 (Small) 로드
        print(f"[DepthEstimator] Loading depth-anything/Depth-Anything-V2-Small-hf on {self.device}...")
        self.pipe = hf_pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Small-hf", 
            device=0 if self.device == "cuda" else -1
        )

    def generate_depth_map(self, image_path: str) -> Image.Image:
        """단일 이미지에 대해 흑백 Depth 맵을 반환합니다."""
        orig_img = Image.open(image_path).convert("RGB")
        
        # 추론
        result = self.pipe(orig_img)
        
        # 결과 반환 (dict 내 'depth' key에 PIL Image로 반환됨)
        depth_img = result["depth"]
        return depth_img


class VisionPipeline:
    """
    RMBG와 Depth 모델을 순차적으로 통과시키는 파이프라인
    """
    def __init__(self, device: str = None):
        self.bg_extractor = RMBGExtractor(device)
        self.depth_estimator = DepthEstimator(device)

    def initialize(self):
        self.bg_extractor.load_model()
        self.depth_estimator.load_model()

    def process(self, image_path: str, output_dir: str):
        """이미지를 처리하고 결과를 output_dir에 저장합니다."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path).split('.')[0]
        
        # 1. 배경 제거
        print(f"-> Extracting background for {base_name}...")
        rgba_img, mask_img = self.bg_extractor.extract_mask(image_path)
        
        # 임시 저장 경로
        rgba_out = os.path.join(output_dir, f"{base_name}_rgba.png")
        mask_out = os.path.join(output_dir, f"{base_name}_mask.png")
        rgba_img.save(rgba_out)
        mask_img.save(mask_out)
        print(f"   Saved Mask: {mask_out}")

        # 2. 깊이 추정
        # Depth는 노이즈(배경)가 제거된 투명 배경 사진(RGBA 중 RGB만)을 기반으로 추출
        print(f"-> Generating depth map for {base_name}...")
        # 투명 배경을 검은색으로 치환하여 Depth 모델에 넣는 것이 더 안정적임
        black_bg = Image.new("RGB", rgba_img.size, (0, 0, 0))
        black_bg.paste(rgba_img, mask=mask_img)
        temp_rgb_path = os.path.join(output_dir, "temp_for_depth.png")
        black_bg.save(temp_rgb_path)

        depth_img = self.depth_estimator.generate_depth_map(temp_rgb_path)
        
        depth_out = os.path.join(output_dir, f"{base_name}_depth.png")
        depth_img.save(depth_out)
        os.remove(temp_rgb_path)
        print(f"   Saved Depth: {depth_out}")
        
        return rgba_out, mask_out, depth_out

if __name__ == '__main__':
    # Test script for single image
    pipeline = VisionPipeline()
    pipeline.initialize()
    
    # 1번 샘플 (TestData의 첫 번째 폴더)
    test_image_path = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData\50073721_5995-37-501-9359\Cable.PNG"
    output_directory = r"C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\ai_pipeline\test_output"
    
    pipeline.process(test_image_path, output_directory)
    print("Done!")
