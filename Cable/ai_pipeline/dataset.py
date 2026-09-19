import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict
from abc import ABC, abstractmethod


@dataclass
class CableDataModel:
    """
    단일 케이블 데이터셋의 경로와 메타데이터를 담는 구조체.
    """
    cable_id: str             # 폴더명 (예: 20102643_5995-37-507-1550)
    root_path: str            # 폴더 절대 경로
    png_path: Optional[str]   # Cable.PNG (또는 유사 이미지) 경로
    pdf_path: Optional[str]   # Cable.pdf 경로
    tcl_path: Optional[str]   # cable.tcl 경로
    is_complete: bool         # PNG, PDF, TCL 3개가 모두 존재하는지 여부

    def to_dict(self) -> Dict:
        return {
            "cable_id": self.cable_id,
            "root_path": self.root_path,
            "png_path": self.png_path,
            "pdf_path": self.pdf_path,
            "tcl_path": self.tcl_path,
            "is_complete": self.is_complete
        }


class CableDatasetBase(ABC):
    """
    케이블 데이터셋 입출력을 위한 추상 기본 클래스 (Interface)
    """
    @abstractmethod
    def list_cables(self) -> List[CableDataModel]:
        pass

    @abstractmethod
    def get_cable(self, cable_id: str) -> Optional[CableDataModel]:
        pass


class LocalCableDataset(CableDatasetBase):
    """
    로컬 디렉토리(TestData, TestData2)를 순회하며 파일을 매핑하는 클래스
    """
    def __init__(self, target_directories: List[str]):
        self.target_directories = target_directories
        self._cables_cache: Dict[str, CableDataModel] = {}
        self._scan_directories()

    def _scan_directories(self):
        """지정된 폴더들을 순회하며 케이블 데이터를 캐싱합니다."""
        for directory in self.target_directories:
            if not os.path.exists(directory):
                print(f"[Warning] Directory not found: {directory}")
                continue
                
            for folder_name in os.listdir(directory):
                folder_path = os.path.join(directory, folder_name)
                if not os.path.isdir(folder_path):
                    continue
                    
                cable_model = self._parse_cable_folder(folder_name, folder_path)
                self._cables_cache[cable_model.cable_id] = cable_model

    def _parse_cable_folder(self, cable_id: str, folder_path: str) -> CableDataModel:
        """단일 폴더 내의 파일들을 분석하여 CableDataModel을 생성합니다."""
        files = os.listdir(folder_path)
        files_lower = [f.lower() for f in files]
        
        png_path = None
        pdf_path = None
        tcl_path = None
        
        # 1. 우선적으로 표준 이름 'Cable.PNG', 'Cable.pdf', 'cable.tcl' 탐색
        for f in files:
            lower_f = f.lower()
            if lower_f == 'cable.png':
                png_path = os.path.join(folder_path, f)
            elif lower_f == 'cable.pdf':
                pdf_path = os.path.join(folder_path, f)
            elif lower_f == 'cable.tcl':
                tcl_path = os.path.join(folder_path, f)

        # 2. 표준 이름이 없으면 해당 확장자를 가진 아무 파일이나 맵핑 (Fallback)
        if not png_path:
            for f in files:
                if f.lower().endswith('.png') or f.lower().endswith('.jpg'):
                    png_path = os.path.join(folder_path, f)
                    break
                    
        if not pdf_path:
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdf_path = os.path.join(folder_path, f)
                    break
                    
        if not tcl_path:
            for f in files:
                if f.lower().endswith('.tcl'):
                    tcl_path = os.path.join(folder_path, f)
                    break
                    
        is_complete = all([png_path, pdf_path, tcl_path])
        
        return CableDataModel(
            cable_id=cable_id,
            root_path=folder_path,
            png_path=png_path,
            pdf_path=pdf_path,
            tcl_path=tcl_path,
            is_complete=is_complete
        )

    def list_cables(self) -> List[CableDataModel]:
        """모든 케이블 데이터 리스트 반환"""
        return list(self._cables_cache.values())

    def get_cable(self, cable_id: str) -> Optional[CableDataModel]:
        """ID로 특정 케이블 데이터 반환"""
        return self._cables_cache.get(cable_id)

    def print_summary(self):
        """데이터셋 현황 요약 출력"""
        total = len(self._cables_cache)
        complete = sum(1 for c in self._cables_cache.values() if c.is_complete)
        print("="*50)
        print(f" Dataset Summary (Local Directories)")
        print("="*50)
        print(f" Total Cables Found : {total}")
        print(f" Complete Sets (PNG+PDF+TCL) : {complete}")
        print(f" Incomplete Sets : {total - complete}")
        
        if total - complete > 0:
            print("\n[Incomplete Cables]")
            for c in self._cables_cache.values():
                if not c.is_complete:
                    missing = []
                    if not c.png_path: missing.append("Image(PNG/JPG)")
                    if not c.pdf_path: missing.append("PDF")
                    if not c.tcl_path: missing.append("TCL")
                    print(f" - {c.cable_id}: Missing {', '.join(missing)}")
        print("="*50)


if __name__ == '__main__':
    # 테스트 코드
    target_dirs = [
        r'C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData',
        r'C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData2'
    ]
    dataset = LocalCableDataset(target_dirs)
    dataset.print_summary()
