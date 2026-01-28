#!/usr/bin/env python3
"""
快速DICOM-JSON配對驗證工具
Quick DICOM-JSON Pair Validation Tool

簡化版驗證工具，快速檢查檔案配對關係
"""

import os
import json
import pydicom
from pathlib import Path

def quick_validate(dicom_path, json_path):
    """快速驗證DICOM與JSON檔案配對"""
    print(f"🔍 驗證檔案配對:")
    print(f"   DICOM: {dicom_path}")
    print(f"   JSON:  {json_path}")
    print()
    
    # 檢查檔案存在性
    if not os.path.exists(dicom_path):
        print("❌ DICOM檔案不存在")
        return False
    
    if not os.path.exists(json_path):
        print("❌ JSON檔案不存在")
        return False
    
    print("✅ 檔案存在性檢查通過")
    
    # 讀取DICOM檔案
    try:
        dicom = pydicom.dcmread(dicom_path)
        print("✅ DICOM檔案格式有效")
    except Exception as e:
        print(f"❌ DICOM檔案讀取失敗: {e}")
        return False
    
    # 讀取JSON檔案
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print("✅ JSON檔案格式有效")
    except Exception as e:
        print(f"❌ JSON檔案讀取失敗: {e}")
        return False
    
    # 檢查基本資訊匹配
    matches = 0
    total_checks = 0
    
    # 1. 患者ID匹配
    total_checks += 1
    dicom_patient_id = getattr(dicom, 'PatientID', '')
    json_patient_id = json_data.get('metadata', {}).get('patient_id', '')
    
    if dicom_patient_id and json_patient_id:
        if dicom_patient_id == json_patient_id:
            print("✅ 患者ID匹配")
            matches += 1
        else:
            print(f"❌ 患者ID不匹配: DICOM={dicom_patient_id}, JSON={json_patient_id}")
    else:
        print("⚠️  無法比較患者ID（缺少資訊）")
    
    # 2. 檢查日期匹配
    total_checks += 1
    dicom_study_date = getattr(dicom, 'StudyDate', '')
    json_study_date = json_data.get('metadata', {}).get('study_date', '')
    
    if dicom_study_date and json_study_date:
        # 正規化日期格式
        if len(dicom_study_date) == 8:
            dicom_date = f"{dicom_study_date[:4]}-{dicom_study_date[4:6]}-{dicom_study_date[6:8]}"
        else:
            dicom_date = dicom_study_date
        
        if dicom_date == json_study_date:
            print("✅ 檢查日期匹配")
            matches += 1
        else:
            print(f"❌ 檢查日期不匹配: DICOM={dicom_date}, JSON={json_study_date}")
    else:
        print("⚠️  無法比較檢查日期（缺少資訊）")
    
    # 3. 影像尺寸匹配
    total_checks += 1
    dicom_rows = getattr(dicom, 'Rows', 0)
    dicom_cols = getattr(dicom, 'Columns', 0)
    json_dims = json_data.get('metadata', {}).get('image_dimensions', {})
    json_width = json_dims.get('width', 0)
    json_height = json_dims.get('height', 0)
    
    if dicom_rows and dicom_cols and json_width and json_height:
        if dicom_rows == json_height and dicom_cols == json_width:
            print("✅ 影像尺寸匹配")
            matches += 1
        else:
            print(f"❌ 影像尺寸不匹配: DICOM={dicom_cols}x{dicom_rows}, JSON={json_width}x{json_height}")
    else:
        print("⚠️  無法比較影像尺寸（缺少資訊）")
    
    # 4. 檢查標註資料
    total_checks += 1
    measurements = json_data.get('measurements', [])
    if measurements:
        print(f"✅ 找到 {len(measurements)} 個椎間隙標註")
        matches += 1
    else:
        print("❌ 沒有找到椎間隙標註")
    
    # 計算匹配度
    match_percentage = (matches / total_checks) * 100
    print()
    print(f"📊 匹配度: {match_percentage:.1f}% ({matches}/{total_checks})")
    
    if match_percentage >= 80:
        print("🎉 檔案配對品質良好，可以安全使用")
        return True
    elif match_percentage >= 60:
        print("⚠️  檔案配對品質一般，建議檢查不匹配的項目")
        return False
    else:
        print("❌ 檔案配對品質較差，建議重新檢查檔案來源")
        return False

def find_matching_json(dicom_path):
    """為DICOM檔案尋找對應的JSON檔案"""
    dicom_file = Path(dicom_path)
    
    # 可能的JSON檔案名稱模式
    possible_names = [
        dicom_file.with_suffix('.json'),
        dicom_file.parent / f"{dicom_file.stem}_annotations.json",
        dicom_file.parent / f"{dicom_file.stem}_measurements.json",
        dicom_file.parent / f"{dicom_file.stem}_spine.json"
    ]
    
    for json_path in possible_names:
        if json_path.exists():
            return str(json_path)
    
    return None

def batch_validate_directory(directory_path):
    """批量驗證目錄中的檔案"""
    directory = Path(directory_path)
    dicom_files = list(directory.glob('**/*.dcm')) + list(directory.glob('**/*.DCM'))
    
    print(f"🔍 在目錄中發現 {len(dicom_files)} 個DICOM檔案")
    print()
    
    valid_pairs = 0
    total_pairs = 0
    
    for dicom_file in dicom_files:
        json_file = find_matching_json(dicom_file)
        
        if json_file:
            total_pairs += 1
            print(f"📁 處理: {dicom_file.name}")
            
            if quick_validate(str(dicom_file), json_file):
                valid_pairs += 1
            print("-" * 50)
        else:
            print(f"⚠️  找不到對應的JSON檔案: {dicom_file.name}")
    
    print()
    print(f"📊 批量驗證結果:")
    print(f"   總配對數: {total_pairs}")
    print(f"   有效配對: {valid_pairs}")
    print(f"   成功率: {(valid_pairs/total_pairs*100) if total_pairs > 0 else 0:.1f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python quick_validate.py <dicom_file> [json_file]")
        print("  python quick_validate.py --batch <directory>")
        print()
        print("範例:")
        print("  python quick_validate.py patient001.dcm")
        print("  python quick_validate.py patient001.dcm patient001.json")
        print("  python quick_validate.py --batch /path/to/dicom/folder")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        # 批量驗證模式
        directory = sys.argv[2]
        batch_validate_directory(directory)
    else:
        # 單一檔案驗證模式
        dicom_file = sys.argv[1]
        json_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not json_file:
            # 自動尋找對應的JSON檔案
            json_file = find_matching_json(dicom_file)
            if not json_file:
                print(f"❌ 找不到對應的JSON檔案: {dicom_file}")
                sys.exit(1)
            print(f"🔍 自動找到JSON檔案: {json_file}")
        
        quick_validate(dicom_file, json_file)
