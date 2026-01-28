#!/usr/bin/env python3
"""
快速測試 - 簡化版
直接在當前資料夾執行，不需要路徑配置
"""

import json
from pathlib import Path

print("🔍 快速測試開始...")
print("=" * 60)

# 1. 檢查當前目錄
current_dir = Path.cwd()
print(f"📁 當前目錄: {current_dir.name}")

# 2. 搜尋JSON檔案
json_files = list(Path('.').glob('**/*.json'))
print(f"\n找到 {len(json_files)} 個JSON檔案")

# 3. 驗證函數
def is_valid(data):
    """簡單驗證"""
    if 'measurements' not in data or 'image_dimensions' not in data:
        return False
    if not isinstance(data['measurements'], list) or len(data['measurements']) == 0:
        return False
    for m in data['measurements']:
        if 'lowerEndplate' not in m or 'upperEndplate' not in m:
            return False
        if len(m.get('lowerEndplate', [])) < 2 or len(m.get('upperEndplate', [])) < 2:
            return False
    return True

# 4. 檢查每個檔案
valid_files = []
print("\n" + "=" * 60)

for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if is_valid(data):
            valid_files.append(json_file)
            levels = len(data['measurements'])
            img_dims = data.get('image_dimensions', {})
            print(f"✅ {json_file.name}")
            print(f"   椎間隙: {levels}, 影像: {img_dims.get('width')}x{img_dims.get('height')}")
    
    except Exception as e:
        print(f"❌ {json_file.name}: {e}")

# 5. 結果
print("\n" + "=" * 60)
print(f"📊 結果:")
print(f"   總檔案數: {len(json_files)}")
print(f"   有效檔案: {len(valid_files)}")

if len(valid_files) > 0:
    print(f"\n✅ 成功！找到 {len(valid_files)} 個可用的標註檔案")
    print(f"\n📋 有效檔案清單:")
    for f in valid_files:
        print(f"   - {f}")
    
    print(f"\n🚀 下一步:")
    print(f"   1. 確認這些檔案有對應的 .dcm 影像檔")
    print(f"   2. 執行: python train_endplate_model.py")
else:
    print(f"\n❌ 未找到有效檔案")
    print(f"\n🔧 可能原因:")
    print(f"   1. JSON 缺少 'measurements' 或 'image_dimensions'")
    print(f"   2. measurements 是空的")
    print(f"   3. 終板點數不足（需要至少2點）")

print("\n" + "=" * 60)

