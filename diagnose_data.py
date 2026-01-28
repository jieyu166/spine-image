#!/usr/bin/env python3
"""
診斷數據問題
Diagnose Data Issues

檢查為什麼找不到標註檔案
"""

import os
import json
from pathlib import Path

def diagnose():
    """診斷數據問題"""
    print("🔍 診斷數據問題...")
    print("=" * 60)
    
    # 1. 檢查當前目錄
    current_dir = Path.cwd()
    print(f"\n📁 當前目錄: {current_dir}")
    
    # 2. 檢查Spine資料夾
    spine_dir = Path("0. Inbox/Spine")
    if not spine_dir.exists():
        print(f"❌ Spine資料夾不存在: {spine_dir}")
        # 嘗試其他可能的路徑
        spine_dir = Path(".")
        print(f"   使用當前目錄: {spine_dir}")
    else:
        print(f"✅ Spine資料夾存在: {spine_dir}")
    
    # 3. 搜尋所有JSON檔案
    print(f"\n🔎 搜尋JSON檔案...")
    json_files = list(spine_dir.glob('**/*.json'))
    print(f"找到 {len(json_files)} 個JSON檔案:")
    
    for json_file in json_files:
        print(f"\n📄 檔案: {json_file}")
        print(f"   路徑: {json_file.absolute()}")
        print(f"   大小: {json_file.stat().st_size} bytes")
        
        # 4. 嘗試讀取並驗證
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   ✅ JSON格式正確")
            
            # 檢查欄位
            has_measurements = 'measurements' in data
            has_dimensions = 'image_dimensions' in data
            
            print(f"   - measurements: {'✅' if has_measurements else '❌'}")
            print(f"   - image_dimensions: {'✅' if has_dimensions else '❌'}")
            
            if has_measurements:
                measurements = data['measurements']
                print(f"   - 椎間隙數量: {len(measurements)}")
                
                # 檢查每個measurement
                for i, m in enumerate(measurements):
                    has_lower = 'lowerEndplate' in m
                    has_upper = 'upperEndplate' in m
                    
                    if not has_lower or not has_upper:
                        print(f"   ⚠️ measurement {i}: lowerEndplate={has_lower}, upperEndplate={has_upper}")
                    else:
                        lower_len = len(m['lowerEndplate'])
                        upper_len = len(m['upperEndplate'])
                        if lower_len < 2 or upper_len < 2:
                            print(f"   ⚠️ measurement {i}: lower點數={lower_len}, upper點數={upper_len}")
            
            # 驗證是否有效
            is_valid = validate_annotation(data)
            print(f"   總體驗證: {'✅ 有效' if is_valid else '❌ 無效'}")
            
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON解析錯誤: {e}")
        except Exception as e:
            print(f"   ❌ 讀取錯誤: {e}")
    
    # 5. 總結
    print("\n" + "=" * 60)
    print("📊 診斷總結:")
    print(f"   總JSON檔案數: {len(json_files)}")
    
    # 重新驗證
    valid_count = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if validate_annotation(data):
                valid_count += 1
        except:
            pass
    
    print(f"   有效標註檔案: {valid_count}")
    
    if valid_count == 0 and len(json_files) > 0:
        print("\n🔧 可能的問題:")
        print("   1. JSON檔案缺少必要欄位")
        print("   2. measurements 或 image_dimensions 格式不正確")
        print("   3. 終板點數不足（需要至少2個點）")
    
    # 6. 顯示範例有效JSON格式
    if valid_count == 0:
        print("\n📝 有效JSON格式範例:")
        example = {
            "measurements": [
                {
                    "level": "L4/L5",
                    "lowerEndplate": [
                        {"x": 100, "y": 200},
                        {"x": 150, "y": 205}
                    ],
                    "upperEndplate": [
                        {"x": 105, "y": 250},
                        {"x": 155, "y": 255}
                    ]
                }
            ],
            "image_dimensions": {
                "width": 1024,
                "height": 768
            }
        }
        print(json.dumps(example, indent=2, ensure_ascii=False))

def validate_annotation(data):
    """驗證標註資料格式（與prepare_endplate_data.py相同）"""
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

if __name__ == "__main__":
    diagnose()
