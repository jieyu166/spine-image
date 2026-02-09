#!/usr/bin/env python3
"""
測試推理並顯示詳細調試資訊 (V3 版)
Test Inference with Debug Info (V3 heatmap model)
"""

from inference_vertebra import VertebraInference
import sys

def main():
    # 您的測試檔案 (支援 DICOM / PNG / JPG)
    test_file = r"C:\Users\jai16\OneDrive\00 放射科\5工作\Spine data\19336656_20251007\19336656.dcm"

    print("=" * 60)
    print("推理調試測試 (V3 Heatmap Model)")
    print("=" * 60)

    # 初始化分析器
    print("\n1. 載入模型...")
    analyzer = VertebraInference('best_vertebra_model.pth', device='auto')

    print("\n2. 執行推理...")
    result = analyzer.predict(test_file, spine_type='L')

    print("\n" + "=" * 60)
    print("最終結果:")
    print("=" * 60)
    print(f"預測椎體數量: {result['predicted_count']}")
    print(f"計數信心度: {result['count_confidence']:.1%}")
    print(f"影像尺寸: {result['image_info']['width']} x {result['image_info']['height']}")

    print("\n椎體角點座標:")
    for v in result['vertebrae']:
        bt_label = f" [{v['boundaryType']}]" if v['boundaryType'] else ""
        print(f"\n  {v['name']}{bt_label}:")
        for corner_name, coord in v['points'].items():
            conf = v.get('confidences', {}).get(corner_name, 0)
            print(f"    {corner_name}: ({coord['x']:.1f}, {coord['y']:.1f}) conf={conf:.1%}")

        if v.get('anteriorHeight') is not None:
            print(f"    前緣高度: {v['anteriorHeight']:.1f}")
            print(f"    後緣高度: {v['posteriorHeight']:.1f}")
            print(f"    高度比: {v['heightRatio']:.2f}")
            if v.get('anteriorWedgingFracture'):
                print(f"    !! 疑似前緣壓迫性骨折")

    print("\n椎間盤分析:")
    for d in result['discs']:
        print(f"  {d['level']}: 前緣={d['anteriorHeight']:.1f} 後緣={d['posteriorHeight']:.1f} 中間={d['middleHeight']:.1f}")

    print("\n3. 儲存視覺化結果...")
    analyzer.visualize(result, output_path='test_inference_debug_output.png')

    print("\n" + "=" * 60)
    print("✅ 測試完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
