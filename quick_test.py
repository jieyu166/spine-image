#!/usr/bin/env python3
"""
快速測試 - 驗證JSON標註檔案
支援 V1 (終板格式) 及 V2.x (椎體頂點格式)
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
def is_valid_v1(data):
    """驗證 V1 格式 (終板)"""
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

def is_valid_v2(data):
    """驗證 V2.x 格式 (椎體頂點)"""
    if 'vertebrae' not in data:
        return False
    vertebrae = data['vertebrae']
    if not isinstance(vertebrae, list) or len(vertebrae) == 0:
        return False
    for v in vertebrae:
        if 'points' not in v:
            return False
        points = v['points']
        if isinstance(points, dict):
            # 至少要有 2 個頂點
            if len(points) < 2:
                return False
        elif isinstance(points, list):
            if len(points) < 2:
                return False
        else:
            return False
    return True

def detect_format(data):
    """偵測格式版本"""
    version = data.get('version', '')
    if isinstance(version, str) and version.startswith('2'):
        return 'v2'
    if 'vertebrae' in data:
        return 'v2'
    if 'measurements' in data:
        return 'v1'
    return None

# 4. 檢查每個檔案
valid_files = []
v1_count = 0
v2_count = 0
skipped = []
print("\n" + "=" * 60)

for json_file in json_files:
    # 跳過非標註的 JSON
    if any(skip in str(json_file) for skip in [
        'training_data', 'dataset_info', 'annotation_template',
        'inference_results', '.claude', '_result'
    ]):
        continue

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        fmt = detect_format(data)

        if fmt == 'v1' and is_valid_v1(data):
            valid_files.append(json_file)
            v1_count += 1
            levels = len(data['measurements'])
            img_dims = data.get('image_dimensions', {})
            # 找對應影像
            img_ext = '—'
            for ext in ['.dcm', '.png', '.jpg']:
                if json_file.with_suffix(ext).exists():
                    img_ext = ext
                    break
            print(f"✅ {json_file.name}  [V1] 椎間隙: {levels}, "
                  f"影像: {img_dims.get('width')}x{img_dims.get('height')}, "
                  f"圖檔: {img_ext}")

        elif fmt == 'v2' and is_valid_v2(data):
            valid_files.append(json_file)
            v2_count += 1
            verts = len(data['vertebrae'])
            img_info = data.get('imageInfo', {})
            spine_type = data.get('spineType', '?')
            # 找對應影像
            img_ext = '—'
            for ext in ['.png', '.jpg', '.dcm']:
                if json_file.with_suffix(ext).exists():
                    img_ext = ext
                    break
            print(f"✅ {json_file.name}  [V2] {spine_type}-spine, 椎體: {verts}, "
                  f"影像: {img_info.get('width')}x{img_info.get('height')}, "
                  f"圖檔: {img_ext}")

        else:
            skipped.append(json_file.name)

    except Exception as e:
        print(f"❌ {json_file.name}: {e}")

# 5. 結果
print("\n" + "=" * 60)
print(f"📊 結果:")
print(f"   總 JSON 檔案數: {len(json_files)}")
print(f"   有效標註檔案: {len(valid_files)}  (V1: {v1_count}, V2: {v2_count})")

if skipped:
    print(f"   略過（非標註）: {len(skipped)}")

if len(valid_files) > 0:
    print(f"\n✅ 成功！找到 {len(valid_files)} 個可用的標註檔案")
    print(f"\n📋 有效檔案清單:")
    for f in valid_files:
        print(f"   - {f}")

    print(f"\n🚀 下一步:")
    print(f"   1. 確認這些檔案有對應的影像檔 (.dcm / .png)")
    print(f"   2. 執行: python prepare_endplate_data.py")
else:
    print(f"\n❌ 未找到有效檔案")
    print(f"\n🔧 可能原因:")
    print(f"   V1: JSON 缺少 'measurements' 或 'image_dimensions'")
    print(f"   V2: JSON 缺少 'vertebrae' 或頂點數不足")

print("\n" + "=" * 60)
