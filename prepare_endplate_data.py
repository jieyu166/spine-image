#!/usr/bin/env python3
"""
終板檢測數據準備腳本
Endplate Detection Data Preparation Script

處理新的JSON標註格式，準備終板檢測訓練數據
"""

import os
import json
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

class EndplateDataPreparer:
    """終板檢測數據準備器"""
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建輸出目錄結構
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'annotations').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
    
    def collect_annotations(self):
        """收集所有標註檔案"""
        print("🔍 收集標註檔案...")
        
        json_files = list(self.input_dir.glob('**/*.json'))
        annotations = []
        
        for json_file in tqdm(json_files, desc="處理JSON檔案"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 驗證必要欄位
                if self.validate_annotation(data):
                    annotations.append({
                        'file': json_file,
                        'data': data
                    })
                else:
                    print(f"⚠️ 跳過無效檔案: {json_file}")
            
            except Exception as e:
                print(f"❌ 讀取檔案失敗 {json_file}: {e}")
        
        print(f"✅ 找到 {len(annotations)} 個有效標註檔案")
        return annotations
    
    def validate_annotation(self, data):
        """驗證標註資料格式"""
        # 檢查必要欄位
        required_fields = ['measurements', 'image_dimensions']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # 檢查measurements內容
        if not isinstance(data['measurements'], list) or len(data['measurements']) == 0:
            return False
        
        # 檢查每個measurement是否包含終板資訊
        for m in data['measurements']:
            if 'lowerEndplate' not in m or 'upperEndplate' not in m:
                return False
            if len(m['lowerEndplate']) < 2 or len(m['upperEndplate']) < 2:
                return False
        
        return True
    
    def prepare_training_data(self, annotations):
        """準備訓練數據"""
        print("📝 準備訓練數據...")
        
        train_annotations = []
        val_annotations = []
        
        for i, ann in enumerate(tqdm(annotations, desc="處理標註")):
            data = ann['data']
            json_file = ann['file']
            
            # 自動尋找對應的DICOM檔案
            image_path = data.get('image_path', '')
            if not image_path or not os.path.exists(image_path):
                # 嘗試找同名的.dcm檔案
                json_path = Path(json_file)
                dcm_path = json_path.with_suffix('.dcm')
                
                if dcm_path.exists():
                    image_path = str(dcm_path.relative_to(self.input_dir))
                else:
                    # 嘗試在同目錄下找同名dcm
                    base_name = json_path.stem
                    dcm_candidates = list(json_path.parent.glob(f'{base_name}*.dcm'))
                    if dcm_candidates:
                        image_path = str(dcm_candidates[0].relative_to(self.input_dir))
            
            # 提取資訊
            processed_data = {
                'patient_id': data.get('patient_id', ''),
                'study_id': data.get('study_id', ''),
                'study_date': data.get('study_date', ''),
                'spine_type': data.get('spine_type', 'L'),
                'image_type': data.get('image_type', 'neutral'),
                'image_path': image_path,
                'image_dimensions': data.get('image_dimensions', {}),
                'annotator': data.get('annotator', {}),
                'annotation_date': data.get('annotation_date', ''),
                'measurements': [],
                'vertebra_edges': data.get('vertebra_edges', {}),
                'clinical_notes': data.get('clinical_notes', {}),
                'surgery_info': data.get('surgery_info', {})
            }
            
            # 處理measurements
            for m in data['measurements']:
                measurement = {
                    'level': m.get('level', ''),
                    'lowerEndplate': m.get('lowerEndplate', []),
                    'upperEndplate': m.get('upperEndplate', []),
                    'confidence': m.get('confidence', 0.95),
                    'measurement_method': m.get('measurement_method', 'manual')
                }
                
                # 可選: 保留角度資訊供參考
                if 'angle' in m:
                    measurement['angle'] = m['angle']
                if 'angle_raw' in m:
                    measurement['angle_raw'] = m['angle_raw']
                
                processed_data['measurements'].append(measurement)
            
            # 80-20分割
            if i % 5 == 0:
                val_annotations.append(processed_data)
            else:
                train_annotations.append(processed_data)
        
        # 保存標註檔案
        train_file = self.output_dir / 'annotations' / 'train_annotations.json'
        val_file = self.output_dir / 'annotations' / 'val_annotations.json'
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_annotations, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_annotations, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 訓練集: {len(train_annotations)} 個樣本 -> {train_file}")
        print(f"✅ 驗證集: {len(val_annotations)} 個樣本 -> {val_file}")
        
        return train_annotations, val_annotations
    
    def analyze_dataset(self, train_annotations, val_annotations):
        """分析數據集統計資訊"""
        print("\n📊 數據集分析:")
        
        all_annotations = train_annotations + val_annotations
        
        # 總樣本數
        print(f"  總樣本數: {len(all_annotations)}")
        print(f"  訓練樣本: {len(train_annotations)}")
        print(f"  驗證樣本: {len(val_annotations)}")
        
        # 脊椎類型統計
        spine_types = {}
        for ann in all_annotations:
            spine_type = ann.get('spine_type', 'L')
            spine_types[spine_type] = spine_types.get(spine_type, 0) + 1
        print(f"\n  脊椎類型分布: {spine_types}")
        
        # 影像類型統計
        image_types = {}
        for ann in all_annotations:
            image_type = ann.get('image_type', 'neutral')
            image_types[image_type] = image_types.get(image_type, 0) + 1
        print(f"  影像類型分布: {image_types}")
        
        # 椎間隙統計
        total_measurements = sum(len(ann['measurements']) for ann in all_annotations)
        avg_measurements = total_measurements / len(all_annotations)
        print(f"\n  總椎間隙數: {total_measurements}")
        print(f"  平均每張影像: {avg_measurements:.1f} 個椎間隙")
        
        # 椎間隙層級統計
        level_counts = {}
        for ann in all_annotations:
            for m in ann['measurements']:
                level = m.get('level', '')
                level_counts[level] = level_counts.get(level, 0) + 1
        
        print(f"\n  椎間隙層級分布:")
        for level in sorted(level_counts.keys()):
            print(f"    {level}: {level_counts[level]}")
        
        # 終板點數統計
        total_endplate_points = 0
        for ann in all_annotations:
            for m in ann['measurements']:
                total_endplate_points += len(m.get('lowerEndplate', [])) + len(m.get('upperEndplate', []))
        print(f"\n  總終板點數: {total_endplate_points}")
        
        # 椎體邊緣統計
        vertebra_with_edges = sum(1 for ann in all_annotations if ann.get('vertebra_edges'))
        total_edges = sum(len(ann.get('vertebra_edges', {})) for ann in all_annotations)
        print(f"\n  包含椎體邊緣的樣本: {vertebra_with_edges}")
        print(f"  總椎體邊緣數: {total_edges}")
        
        # 手術資訊統計
        surgery_cases = sum(1 for ann in all_annotations 
                          if ann.get('surgery_info', {}).get('surgery_done', False))
        print(f"\n  術後病例: {surgery_cases}/{len(all_annotations)}")
        
        # 影像尺寸統計
        widths = [ann['image_dimensions'].get('width', 0) for ann in all_annotations]
        heights = [ann['image_dimensions'].get('height', 0) for ann in all_annotations]
        
        if widths and heights:
            print(f"\n  影像尺寸範圍:")
            print(f"    寬度: {min(widths)} - {max(widths)} (平均: {np.mean(widths):.0f})")
            print(f"    高度: {min(heights)} - {max(heights)} (平均: {np.mean(heights):.0f})")
    
    def create_dataset_info(self):
        """創建數據集資訊檔案"""
        info = {
            "dataset_name": "Spine Endplate Detection Dataset",
            "description": "脊椎終板檢測機器學習數據集 - 專注於終板前後緣檢測",
            "version": "2.0",
            "created_by": "AI Assistant",
            "format": {
                "images": "DICOM/JPG",
                "annotations": "JSON",
                "coordinate_system": "pixel coordinates"
            },
            "tasks": [
                "endplate_segmentation",
                "vertebra_edge_detection",
                "keypoint_detection"
            ],
            "annotation_format": {
                "measurements": [
                    {
                        "level": "椎間隙標籤 (如 L4/L5)",
                        "lowerEndplate": "下終板2個端點座標 [{x, y}, {x, y}]",
                        "upperEndplate": "上終板2個端點座標 [{x, y}, {x, y}]",
                        "confidence": "信心度 (0-1)"
                    }
                ],
                "vertebra_edges": {
                    "vertebra_name": {
                        "anterior": "前緣2個端點座標",
                        "posterior": "後緣2個端點座標"
                    }
                }
            },
            "model_outputs": [
                "endplate_segmentation_mask",
                "vertebra_anterior_edge",
                "vertebra_posterior_edge",
                "endplate_keypoints"
            ],
            "notes": "機器學習模型只負責檢測終板前後緣，不計算角度"
        }
        
        with open(self.output_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        print("\n✅ 創建數據集資訊檔案")
    
    def process_all(self):
        """處理所有數據"""
        print("🚀 開始終板檢測數據準備流程...")
        
        # 1. 收集標註檔案
        annotations = self.collect_annotations()
        
        if not annotations:
            print("❌ 未找到有效的標註檔案")
            return
        
        # 2. 準備訓練數據
        train_annotations, val_annotations = self.prepare_training_data(annotations)
        
        # 3. 分析數據集
        self.analyze_dataset(train_annotations, val_annotations)
        
        # 4. 創建數據集資訊
        self.create_dataset_info()
        
        print("\n🎉 數據準備完成!")
        print(f"📁 輸出目錄: {self.output_dir}")
        print("📋 生成的檔案:")
        print("  - annotations/train_annotations.json")
        print("  - annotations/val_annotations.json")
        print("  - dataset_info.json")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='終板檢測數據準備')
    parser.add_argument('--input_dir', type=str, default='.',
                       help='標註檔案目錄（預設為當前目錄）')
    parser.add_argument('--output_dir', type=str, default='endplate_training_data',
                       help='輸出目錄（預設為當前目錄下的 endplate_training_data）')
    
    args = parser.parse_args()
    
    # 創建數據準備器
    preparer = EndplateDataPreparer(args.input_dir, args.output_dir)
    
    # 處理所有數據
    preparer.process_all()

if __name__ == "__main__":
    main()
