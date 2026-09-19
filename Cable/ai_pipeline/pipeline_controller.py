import os
import trimesh
from dataset import LocalCableDataset
from vision import VisionPipeline
from geometry import CableDigitalTwinBuilder

class PipelineController:
    """
    Data -> Vision -> Geometry -> Export 까지의 모든 흐름을 관장하는 최종 관제탑 클래스
    """
    def __init__(self, target_directories: list[str], output_root_dir: str):
        self.dataset = LocalCableDataset(target_directories)
        self.output_root_dir = output_root_dir
        self.vision_pipeline = VisionPipeline()
        self.is_initialized = False

    def initialize(self):
        print("[PipelineController] Initializing AI Models...")
        self.vision_pipeline.initialize()
        os.makedirs(self.output_root_dir, exist_ok=True)
        self.is_initialized = True
        print("[PipelineController] AI Models Loaded successfully.")

    def run_batch(self):
        """전체 케이블 데이터셋에 대해 3D 디지털 트윈을 일괄 자동 생성합니다."""
        if not self.is_initialized:
            self.initialize()
            
        cables = self.dataset.list_cables()
        total_cables = len(cables)
        success_count = 0
        fail_count = 0
        
        print("="*60)
        print(f"🚀 Starting Batch 3D Digital Twin Generation for {total_cables} Cables")
        print("="*60)
        
        for idx, cable in enumerate(cables):
            print(f"\n[{idx+1}/{total_cables}] Processing Cable: {cable.cable_id}")
            if not cable.is_complete:
                print(f"  -> [Skip] Incomplete dataset (Missing PNG/PDF/TCL)")
                fail_count += 1
                continue
                
            # 케이블별 고유 출력 폴더
            cable_out_dir = os.path.join(self.output_root_dir, cable.cable_id)
            final_glb_path = os.path.join(cable_out_dir, f"{cable.cable_id}_DigitalTwin.glb")
            
            # 이미 생성된 경우 스킵할 수 있는 옵션 (덮어쓰려면 주석 처리)
            # if os.path.exists(final_glb_path):
            #     print(f"  -> [Skip] Already generated: {final_glb_path}")
            #     success_count += 1
            #     continue
                
            try:
                # 1. Vision Layer (마스크 및 Depth 맵 생성)
                _, mask_path, depth_path = self.vision_pipeline.process(cable.png_path, cable_out_dir)

                # 2~3. Geometry Layer - Shell(AI/Depth 기반 외형) + Wire(TCL 실배선) 생성
                print("  -> Building Digital Twin (Shell + TCL Wires)...")
                twin_builder = CableDigitalTwinBuilder(
                    depth_path=depth_path,
                    mask_path=mask_path,
                    tcl_path=cable.tcl_path,
                )
                combined = twin_builder.build()

                # 4. Export
                print("  -> Exporting GLB...")
                combined.export(final_glb_path)
                print(f"  => ✨ Success! Saved to {final_glb_path}")
                success_count += 1
                
                # 용량 절약을 위해 임시 이미지(Mask, Depth) 삭제
                os.remove(mask_path)
                os.remove(depth_path)
                os.remove(os.path.join(cable_out_dir, f"{cable.cable_id}_rgba.png"))
                
            except Exception as e:
                print(f"  -> ❌ [Error] Failed to process {cable.cable_id}: {e}")
                fail_count += 1
                
        print("\n" + "="*60)
        print("🎉 Batch Processing Completed!")
        print(f"✅ Success: {success_count} / {total_cables}")
        print(f"❌ Failed: {fail_count} / {total_cables}")
        print("="*60)


if __name__ == '__main__':
    target_dirs = [
        r'C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData',
        r'C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\TestData2'
    ]
    # 생성된 GLB 파일들이 저장될 최종 결과물 폴더
    output_directory = r'C:\AIND_Library\Git_PredicTest\x64_Cami_Lib\PSCD1500\PSCD1500\Cable\DigitalTwins'
    
    controller = PipelineController(target_dirs, output_directory)
    # 배치 실행
    controller.run_batch()
