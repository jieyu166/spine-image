#!/usr/bin/env python3
"""
檢查DICOM檔案配對
"""

import json
from pathlib import Path

print("🔍 檢查DICOM和JSON配對...")
print("=" * 60)

# 搜尋所有JSON和DICOM檔案
json_files = list(Path('.').glob('**/*.json'))
dcm_files = list(Path('.').glob('**/*.dcm'))

print(f"\n找到 {len(json_files)} 個JSON檔案")
print(f"找到 {len(dcm_files)} 個DICOM檔案")

# 檢查配對
print("\n" + "=" * 60)
print("配對檢查:")
print("=" * 60)

for json_file in json_files:
    # 跳過特定資料夾
    if 'endplate_training_data' in str(json_file):
        continue
    
    print(f"\n📄 {json_file.name}")
    print(f"   路徑: {json_file}")
    
    # 嘗試讀取JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 檢查是否有效
        if 'measurements' not in data:
            print(f"   ⚠️ 不是標註檔案（無measurements）")
            continue
        
        print(f"   椎間隙數: {len(data['measurements'])}")
        
        # 檢查對應的DICOM
        dcm_path = json_file.with_suffix('.dcm')
        
        if dcm_path.exists():
            print(f"   ✅ 找到配對DICOM: {dcm_path.name}")
        else:
            # 搜尋同名DICOM
            base_name = json_file.stem
            found = False
            
            for dcm in dcm_files:
                if dcm.stem.startswith(base_name):
                    print(f"   ✅ 找到相似DICOM: {dcm.name}")
                    found = True
                    break
            
            if not found:
                print(f"   ❌ 找不到對應的DICOM")
    
    except Exception as e:
        print(f"   ❌ 錯誤: {e}")

print("\n" + "=" * 60)
print("📊 摘要:")
print("=" * 60)

# 統計有效配對
valid_pairs = 0
for json_file in json_files:
    if 'endplate_training_data' in str(json_file):
        continue
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'measurements' in data:
            dcm_path = json_file.with_suffix('.dcm')
            if dcm_path.exists():
                valid_pairs += 1
    except:
        pass

print(f"有效的JSON-DICOM配對: {valid_pairs}")
print("\n如果配對數為0，請確認:")
print("1. DICOM檔案和JSON檔案在同一目錄")
print("2. 檔案名稱相同（除了副檔名）")
print("3. 例如: 198261530.json 配對 198261530.dcm")

