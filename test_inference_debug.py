#!/usr/bin/env python3
"""
測試推理並顯示詳細調試資訊
"""

from inference import SpineAnalyzer
import sys

def main():
    # 您的測試檔案
    test_file = r"C:\Users\jai16\OneDrive\00 放射科\5工作\Spine data\19336656_20251007\19336656.dcm"
    
    print("=" * 60)
    print("推理調試測試")
    print("=" * 60)
    
    # 初始化分析器
    print("\n1. 載入模型...")
    analyzer = SpineAnalyzer('best_endplate_model.pth', device='auto')
    
    print("\n2. 執行推理...")
    results = analyzer.predict(test_file)
    
    print("\n3. 提取終板線段...")
    endplate_lines = analyzer.extract_endplates(results['endplate_mask'])
    
    print("\n4. 計算角度...")
    angles = analyzer.calculate_angles(endplate_lines)
    
    print("\n" + "=" * 60)
    print("最終結果:")
    print("=" * 60)
    print(f"檢測到 {len(endplate_lines)} 條終板線段")
    print(f"計算出 {len(angles)} 個角度")
    
    print("\n終板線段座標:")
    for i, line in enumerate(endplate_lines):
        print(f"  線段 {i+1}: ({line[0]}, {line[1]}) -> ({line[2]}, {line[3]})")
    
    print("\n角度計算結果:")
    for angle_info in angles:
        print(f"  椎間隙 #{angle_info['level']}: {angle_info['angle']}°")
    
    print("\n" + "=" * 60)
    print("✅ 測試完成")
    print("=" * 60)

if __name__ == "__main__":
    main()

