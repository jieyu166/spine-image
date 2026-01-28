#!/usr/bin/env python3
"""
快速測試終板檢測模型
Quick Test for Endplate Detection Model

用於測試模型在少量標註數據上的表現
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from train_endplate_model import EndplateDetectionModel, EndplateDataset, get_transforms
import warnings
warnings.filterwarnings('ignore')

class QuickTester:
    """快速測試器"""
    
    def __init__(self, data_dir, device='cpu'):
        self.data_dir = Path(data_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        print(f"🔧 使用設備: {self.device}")
    
    def collect_annotations(self):
        """收集所有標註檔案"""
        print("\n📁 收集標註檔案...")
        
        json_files = list(self.data_dir.glob('**/*.json'))
        valid_annotations = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 檢查必要欄位
                if 'measurements' in data and len(data['measurements']) > 0:
                    valid_annotations.append({
                        'file': json_file,
                        'data': data
                    })
                    print(f"  ✅ {json_file.name}: {len(data['measurements'])} 個椎間隙")
            
            except Exception as e:
                print(f"  ⚠️ 跳過 {json_file.name}: {e}")
        
        print(f"\n✅ 找到 {len(valid_annotations)} 個有效標註檔案")
        return valid_annotations
    
    def create_test_model(self):
        """創建測試模型"""
        print("\n🤖 創建測試模型...")
        
        self.model = EndplateDetectionModel(pretrained=False)  # 不使用預訓練，更快
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 計算參數數量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"  總參數: {total_params:,}")
        print(f"  可訓練參數: {trainable_params:,}")
        print(f"  模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return self.model
    
    def test_forward_pass(self, test_image_path=None):
        """測試前向傳播"""
        print("\n🔬 測試前向傳播...")
        
        # 創建測試圖像
        if test_image_path and os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # 創建隨機測試圖像
            image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            print("  ⚠️ 使用隨機測試圖像")
        
        # 預處理
        transform = get_transforms(is_training=False)
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        print(f"  輸入形狀: {input_tensor.shape}")
        
        # 前向傳播
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # 檢查輸出
        print("\n  輸出檢查:")
        for key, value in outputs.items():
            print(f"    {key}: {value.shape}")
            print(f"      - 範圍: [{value.min():.4f}, {value.max():.4f}]")
            print(f"      - 平均值: {value.mean():.4f}")
        
        return outputs
    
    def visualize_model_output(self, image_path, output_path='test_output.png'):
        """視覺化模型輸出"""
        print(f"\n🎨 視覺化模型輸出...")
        
        if not os.path.exists(image_path):
            print(f"  ❌ 圖像不存在: {image_path}")
            return
        
        # 讀取圖像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]
        
        # 預處理
        transform = get_transforms(is_training=False)
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # 提取輸出
        endplate_seg = outputs['endplate_seg'][0, 0].cpu().numpy()
        vertebra_edge_seg = outputs['vertebra_edge_seg'][0].cpu().numpy()
        keypoint_heatmap = outputs['keypoint_heatmap'][0, 0].cpu().numpy()
        
        # 調整大小回原始尺寸
        endplate_seg = cv2.resize(endplate_seg, (original_size[1], original_size[0]))
        vertebra_edge_ant = cv2.resize(vertebra_edge_seg[0], (original_size[1], original_size[0]))
        vertebra_edge_post = cv2.resize(vertebra_edge_seg[1], (original_size[1], original_size[0]))
        keypoint_heatmap = cv2.resize(keypoint_heatmap, (original_size[1], original_size[0]))
        
        # 創建視覺化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('終板檢測模型輸出 (未訓練)', fontsize=16, fontweight='bold')
        
        # 原始圖像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始圖像')
        axes[0, 0].axis('off')
        
        # 終板分割
        axes[0, 1].imshow(image)
        axes[0, 1].imshow(endplate_seg, alpha=0.5, cmap='jet')
        axes[0, 1].set_title('終板分割遮罩\n(紅色=檢測到的終板)')
        axes[0, 1].axis('off')
        
        # 前緣檢測
        axes[0, 2].imshow(image)
        axes[0, 2].imshow(vertebra_edge_ant, alpha=0.5, cmap='Blues')
        axes[0, 2].set_title('前緣檢測\n(藍色=anterior edge)')
        axes[0, 2].axis('off')
        
        # 後緣檢測
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(vertebra_edge_post, alpha=0.5, cmap='Oranges')
        axes[1, 0].set_title('後緣檢測\n(橙色=posterior edge)')
        axes[1, 0].axis('off')
        
        # 關鍵點熱圖
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(keypoint_heatmap, alpha=0.5, cmap='hot')
        axes[1, 1].set_title('關鍵點熱圖\n(亮點=終板端點)')
        axes[1, 1].axis('off')
        
        # 綜合視圖
        axes[1, 2].imshow(image)
        axes[1, 2].imshow(endplate_seg, alpha=0.3, cmap='Reds')
        axes[1, 2].imshow(vertebra_edge_ant, alpha=0.3, cmap='Blues')
        axes[1, 2].imshow(vertebra_edge_post, alpha=0.3, cmap='Oranges')
        axes[1, 2].set_title('綜合視圖\n(紅=終板, 藍=前緣, 橙=後緣)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 視覺化結果已儲存: {output_path}")
        
        return fig
    
    def quick_training_test(self, annotations, num_epochs=5):
        """快速訓練測試（少量epoch）"""
        print(f"\n🏋️ 快速訓練測試 ({num_epochs} epochs)...")
        
        if len(annotations) == 0:
            print("  ❌ 沒有標註數據，無法訓練")
            return
        
        # 準備數據
        print("  準備訓練數據...")
        train_data = []
        for ann in annotations[:min(len(annotations), 10)]:  # 最多10個樣本
            train_data.append(ann['data'])
        
        # 創建臨時標註檔案
        temp_file = 'temp_annotations.json'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        try:
            # 創建數據集
            transform = get_transforms(is_training=True)
            dataset = EndplateDataset(str(self.data_dir), temp_file, transform=transform)
            
            # 創建數據載入器
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            # 創建優化器和損失函數
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
            from train_endplate_model import EndplateLoss
            criterion = EndplateLoss()
            
            # 訓練
            self.model.train()
            losses = []
            
            for epoch in range(num_epochs):
                epoch_loss = 0
                for batch_idx, (images, targets) in enumerate(loader):
                    images = images.to(self.device)
                    targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in targets.items()}
                    
                    optimizer.zero_grad()
                    predictions = self.model(images)
                    loss_dict = criterion(predictions, targets)
                    
                    loss_dict['total_loss'].backward()
                    optimizer.step()
                    
                    epoch_loss += loss_dict['total_loss'].item()
                
                avg_loss = epoch_loss / len(loader)
                losses.append(avg_loss)
                print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
            
            # 繪製損失曲線
            plt.figure(figsize=(10, 6))
            plt.plot(losses, marker='o')
            plt.title('快速訓練測試 - 損失曲線')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('quick_training_loss.png', dpi=150, bbox_inches='tight')
            print(f"\n  ✅ 損失曲線已儲存: quick_training_loss.png")
            
        finally:
            # 清理臨時檔案
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def generate_test_report(self, annotations):
        """生成測試報告"""
        print("\n📊 生成測試報告...")
        
        report = {
            "test_date": str(Path().absolute()),
            "device": str(self.device),
            "dataset_info": {
                "total_annotations": len(annotations),
                "total_measurements": sum(len(a['data']['measurements']) for a in annotations),
                "spine_types": {},
                "image_types": {}
            },
            "model_info": {
                "architecture": "U-Net with ResNet50",
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
        
        # 統計資訊
        for ann in annotations:
            data = ann['data']
            spine_type = data.get('spine_type', 'Unknown')
            image_type = data.get('image_type', 'Unknown')
            
            report['dataset_info']['spine_types'][spine_type] = \
                report['dataset_info']['spine_types'].get(spine_type, 0) + 1
            report['dataset_info']['image_types'][image_type] = \
                report['dataset_info']['image_types'].get(image_type, 0) + 1
        
        # 儲存報告
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print("  ✅ 測試報告已儲存: test_report.json")
        
        # 列印摘要
        print("\n" + "="*50)
        print("測試摘要")
        print("="*50)
        print(f"標註檔案數: {report['dataset_info']['total_annotations']}")
        print(f"總椎間隙數: {report['dataset_info']['total_measurements']}")
        print(f"脊椎類型: {report['dataset_info']['spine_types']}")
        print(f"影像類型: {report['dataset_info']['image_types']}")
        print(f"模型參數: {report['model_info']['total_parameters']:,}")
        print("="*50)
        
        return report

def main():
    """主函數"""
    print("="*60)
    print("🚀 終板檢測模型快速測試")
    print("="*60)
    
    # 設定路徑
    data_dir = '0. Inbox/Spine'
    
    # 創建測試器
    tester = QuickTester(data_dir, device='cuda')  # 或 'cpu'
    
    # 步驟1: 收集標註
    annotations = tester.collect_annotations()
    
    if len(annotations) == 0:
        print("\n❌ 未找到有效的標註數據，無法繼續測試")
        return
    
    # 步驟2: 創建模型
    tester.create_test_model()
    
    # 步驟3: 測試前向傳播
    tester.test_forward_pass()
    
    # 步驟4: 視覺化輸出（使用第一個標註的圖像）
    first_ann = annotations[0]
    image_path = first_ann['data'].get('image_path', '')
    
    # 嘗試找到對應的DICOM檔案
    if not os.path.exists(image_path):
        # 尋找同名的.dcm檔案
        json_path = first_ann['file']
        dcm_path = json_path.with_suffix('.dcm')
        if dcm_path.exists():
            print(f"\n✅ 找到對應的DICOM檔案: {dcm_path}")
            # 這裡需要將DICOM轉為圖像
        else:
            print(f"\n⚠️ 找不到圖像檔案: {image_path}")
    else:
        tester.visualize_model_output(image_path)
    
    # 步驟5: 快速訓練測試
    response = input("\n是否進行快速訓練測試？(5 epochs) [y/n]: ")
    if response.lower() == 'y':
        tester.quick_training_test(annotations, num_epochs=5)
    
    # 步驟6: 生成報告
    tester.generate_test_report(annotations)
    
    print("\n" + "="*60)
    print("✅ 測試完成！")
    print("="*60)
    print("\n生成的檔案:")
    print("  - test_output.png (模型輸出視覺化)")
    print("  - quick_training_loss.png (訓練損失曲線)")
    print("  - test_report.json (測試報告)")
    print("\n下一步建議:")
    print("  1. 檢查視覺化結果，確認模型架構正確")
    print("  2. 如果損失下降，可以進行完整訓練")
    print("  3. 準備更多標註數據以提升性能")

if __name__ == "__main__":
    main()
