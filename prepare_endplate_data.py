#!/usr/bin/env python3
"""
椎體頂點檢測數據準備腳本 V2.2
Vertebra Corner Detection Data Preparation Script

支援格式:
- V2.2: 每個終板含中點 (完整椎體 6 點, 邊界椎體 3 點)
- V2.1: 邊界椎體僅需 2 點 (S1/T1=上終板, T12/C2=下終板) (向下相容)
- V2.0: 每個椎體 4 個頂點 (向下相容)
- V1.0: 終板格式 (自動轉換)
"""

import os
import json
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse

class VertebraDataPreparer:
    """椎體頂點檢測數據準備器"""

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
            # 跳過配置檔案和訓練資料
            if 'training_data' in str(json_file) or 'dataset_info' in json_file.name:
                continue
            if 'annotation_template' in json_file.name:
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 驗證必要欄位
                version = data.get('version', '1.0')

                if version.startswith('2'):
                    # V2.0 格式（椎體4頂點）
                    if self.validate_v2_annotation(data):
                        annotations.append({
                            'file': json_file,
                            'data': data,
                            'format': 'v2'
                        })
                    else:
                        print(f"⚠️ V2格式無效: {json_file}")
                else:
                    # V1.0 格式（終板格式）- 嘗試轉換
                    if self.validate_v1_annotation(data):
                        annotations.append({
                            'file': json_file,
                            'data': data,
                            'format': 'v1'
                        })
                    else:
                        print(f"⚠️ V1格式無效: {json_file}")

            except Exception as e:
                print(f"❌ 讀取檔案失敗 {json_file}: {e}")

        print(f"✅ 找到 {len(annotations)} 個有效標註檔案")
        return annotations

    # 邊界椎體定義
    BOUNDARY_CONFIG = {
        'L': {'upper': ['S1'], 'lower': ['T12']},   # S1=上終板, T12=下終板
        'C': {'upper': ['T1'], 'lower': ['C2']},     # T1=上終板, C2=下終板
    }

    def is_boundary_vertebra(self, name, spine_type, boundary_type=None, points=None):
        """判斷是否為邊界椎體

        Args:
            name: 椎體名稱 (如 'S1', 'T12')
            spine_type: 脊椎類型 ('L' 或 'C')
            boundary_type: 如果已在 JSON 中指定 (V2.1)，直接使用
            points: 椎體的 points dict，用於判斷 V2.0 是否有完整 4 點

        Returns:
            'upper' (僅上終板), 'lower' (僅下終板), 或 None (完整椎體)
        """
        # V2.1 明確標記的 boundaryType
        if boundary_type:
            return boundary_type

        # 如果有完整 4+ 點 (V2.0/V2.2 格式)，即使名稱是邊界椎體也當完整處理
        if points and isinstance(points, dict):
            has_all_4 = all(k in points for k in
                ['anteriorSuperior', 'posteriorSuperior', 'posteriorInferior', 'anteriorInferior'])
            if has_all_4:
                return None
        elif points and isinstance(points, list) and len(points) >= 4:
            return None

        # 根據名稱和脊椎類型判斷
        config = self.BOUNDARY_CONFIG.get(spine_type, {})
        if name in config.get('upper', []):
            return 'upper'
        if name in config.get('lower', []):
            return 'lower'
        return None

    def validate_v2_annotation(self, data):
        """驗證 V2.0/V2.1/V2.2 標註資料格式（椎體頂點）

        V2.0: 每個椎體 4 點
        V2.1: 邊界椎體可以只有 2 點
        V2.2: 每個終板含中點 (完整椎體 6 點, 邊界椎體 3 點)
        """
        # 檢查必要欄位
        if 'vertebrae' not in data:
            return False

        vertebrae = data['vertebrae']
        if not isinstance(vertebrae, list) or len(vertebrae) == 0:
            return False

        spine_type = data.get('spineType', 'L')

        for v in vertebrae:
            if 'points' not in v:
                return False
            points = v['points']
            name = v.get('name', '')
            bt = v.get('boundaryType', None)
            boundary = self.is_boundary_vertebra(name, spine_type, bt, points)

            if isinstance(points, dict):
                if boundary:
                    # 邊界椎體至少需 2 點 (V2.1) 或 3 點 (V2.2)
                    if boundary == 'upper':
                        required = ['anteriorSuperior', 'posteriorSuperior']
                    else:
                        required = ['posteriorInferior', 'anteriorInferior']
                    if not all(k in points for k in required):
                        return False
                else:
                    # 完整椎體至少需 4 點 (V2.0) 或 6 點 (V2.2)
                    required = ['anteriorSuperior', 'posteriorSuperior',
                               'posteriorInferior', 'anteriorInferior']
                    if not all(k in points for k in required):
                        return False
            elif isinstance(points, list):
                if boundary:
                    if len(points) < 2:
                        return False
                else:
                    if len(points) < 4:
                        return False
            else:
                return False

        return True

    def validate_v1_annotation(self, data):
        """驗證 V1.0 標註資料格式（終板格式）"""
        required_fields = ['measurements', 'image_dimensions']

        for field in required_fields:
            if field not in data:
                return False

        if not isinstance(data['measurements'], list) or len(data['measurements']) == 0:
            return False

        for m in data['measurements']:
            if 'lowerEndplate' not in m or 'upperEndplate' not in m:
                return False
            if len(m['lowerEndplate']) < 2 or len(m['upperEndplate']) < 2:
                return False

        return True

    def convert_v1_to_v2(self, v1_data):
        """將 V1.0 格式轉換為 V2.0 格式"""
        # V1 格式是以椎間盤為中心，需要重建椎體
        # 這是一個近似轉換

        measurements = v1_data['measurements']
        vertebrae = []

        # 從 measurements 提取椎體資訊
        for i, m in enumerate(measurements):
            level = m.get('level', f'Level_{i}')
            upper_name, lower_name = level.split('/')

            upper_endplate = m['upperEndplate']  # 上椎體的下終板
            lower_endplate = m['lowerEndplate']  # 下椎體的上終板

            # 簡化處理：從終板推算椎體頂點（假設椎體高度約為終板長度的 80%）
            # 注意：這是近似值，建議使用 V2 格式重新標註

            # 上椎體（如果是第一個測量）
            if i == 0:
                # 估算上終板位置
                width = abs(upper_endplate[1]['x'] - upper_endplate[0]['x'])
                estimated_height = width * 0.3  # 估算椎體高度

                vertebrae.append({
                    'name': upper_name,
                    'points': {
                        'anteriorSuperior': {
                            'x': upper_endplate[0]['x'],
                            'y': upper_endplate[0]['y'] - estimated_height
                        },
                        'posteriorSuperior': {
                            'x': upper_endplate[1]['x'],
                            'y': upper_endplate[1]['y'] - estimated_height
                        },
                        'posteriorInferior': upper_endplate[1],
                        'anteriorInferior': upper_endplate[0]
                    },
                    'source': 'converted_from_v1'
                })

            # 下椎體
            width = abs(lower_endplate[1]['x'] - lower_endplate[0]['x'])
            estimated_height = width * 0.3

            vertebrae.append({
                'name': lower_name,
                'points': {
                    'anteriorSuperior': lower_endplate[0],
                    'posteriorSuperior': lower_endplate[1],
                    'posteriorInferior': {
                        'x': lower_endplate[1]['x'],
                        'y': lower_endplate[1]['y'] + estimated_height
                    },
                    'anteriorInferior': {
                        'x': lower_endplate[0]['x'],
                        'y': lower_endplate[0]['y'] + estimated_height
                    }
                },
                'source': 'converted_from_v1'
            })

        # 去除重複椎體
        seen = set()
        unique_vertebrae = []
        for v in vertebrae:
            if v['name'] not in seen:
                seen.add(v['name'])
                unique_vertebrae.append(v)

        return {
            'version': '2.0',
            'spineType': v1_data.get('spine_type', 'L'),
            'imageInfo': v1_data.get('image_dimensions', {}),
            'vertebrae': unique_vertebrae,
            'converted_from': 'v1'
        }

    def calculate_vertebra_metrics(self, vertebra, spine_type='L'):
        """計算椎體指標

        邊界椎體 (S1/T1/T12/C2) 只有 2-3 點，無法計算高度比，
        返回 None 用於高度和骨折判斷。
        """
        points = vertebra['points']
        name = vertebra.get('name', '')
        bt = vertebra.get('boundaryType', None)
        boundary = self.is_boundary_vertebra(name, spine_type, bt, points)

        if boundary:
            # 邊界椎體只有 2-3 點，無法計算前後緣高度
            return {
                'anteriorHeight': None,
                'middleHeight': None,
                'posteriorHeight': None,
                'heightRatio': None,
                'compressionFracture': False,
                'boundary': boundary
            }

        # 完整椎體 - 4 點 (V2.0) 或 6 點 (V2.2)
        if isinstance(points, dict):
            ant_sup = points['anteriorSuperior']
            post_sup = points['posteriorSuperior']
            post_inf = points['posteriorInferior']
            ant_inf = points['anteriorInferior']
            mid_sup = points.get('middleSuperior', None)
            mid_inf = points.get('middleInferior', None)
        else:
            if len(points) >= 6:
                # V2.2: [AS, MS, PS, PI, MI, AI]
                ant_sup, mid_sup, post_sup, post_inf, mid_inf, ant_inf = points[:6]
            else:
                # V2.0: [AS, PS, PI, AI]
                ant_sup, post_sup, post_inf, ant_inf = points[:4]
                mid_sup = None
                mid_inf = None

        # 前緣高度
        anterior_height = np.sqrt(
            (ant_inf['x'] - ant_sup['x'])**2 +
            (ant_inf['y'] - ant_sup['y'])**2
        )

        # 後緣高度
        posterior_height = np.sqrt(
            (post_inf['x'] - post_sup['x'])**2 +
            (post_inf['y'] - post_sup['y'])**2
        )

        # 中間高度 (V2.2)
        middle_height = None
        if mid_sup and mid_inf:
            middle_height = np.sqrt(
                (mid_inf['x'] - mid_sup['x'])**2 +
                (mid_inf['y'] - mid_sup['y'])**2
            )

        # 骨折判斷 (cast to Python native types for JSON serialization)
        anterior_wedging = bool(anterior_height < posterior_height * 0.75)
        crush_deformity = bool(anterior_height > posterior_height * 1.25)

        return {
            'anteriorHeight': float(anterior_height),
            'middleHeight': float(middle_height) if middle_height is not None else None,
            'posteriorHeight': float(posterior_height),
            'heightRatio': float(anterior_height / posterior_height) if posterior_height > 0 else 0,
            'compressionFracture': anterior_wedging,  # 向下相容
            'anteriorWedging': anterior_wedging,
            'crushDeformity': crush_deformity,
            'boundary': None
        }

    def get_lower_endplate(self, vertebra, spine_type='L'):
        """取得椎體的下終板 (anterior, [middle], posterior)

        V2.2: 返回 (anterior, middle, posterior) 3 點
        V2.0/V2.1: 返回 (anterior, None, posterior)，middle 為 None

        完整椎體: 取 anteriorInferior + [middleInferior] + posteriorInferior
        下邊界椎體 (T12/C2): 只有下終板 → anteriorInferior + [middleInferior] + posteriorInferior
        上邊界椎體 (S1/T1): 沒有下終板，不應呼叫此方法
        """
        points = vertebra['points']
        bt = vertebra.get('boundaryType', None)
        boundary = self.is_boundary_vertebra(vertebra.get('name', ''), spine_type, bt, points)

        if isinstance(points, dict):
            ant = points.get('anteriorInferior', points.get('anteriorSuperior'))
            post = points.get('posteriorInferior', points.get('posteriorSuperior'))
            mid = points.get('middleInferior', None)
            return ant, mid, post
        else:
            if len(points) >= 6:
                # V2.2: [AS, MS, PS, PI, MI, AI]
                return points[5], points[4], points[3]
            else:
                # V2.0: [AS, PS, PI, AI]
                return points[3], None, points[2]

    def get_upper_endplate(self, vertebra, spine_type='L'):
        """取得椎體的上終板 (anterior, [middle], posterior)

        V2.2: 返回 (anterior, middle, posterior) 3 點
        V2.0/V2.1: 返回 (anterior, None, posterior)，middle 為 None

        完整椎體: 取 anteriorSuperior + [middleSuperior] + posteriorSuperior
        上邊界椎體 (S1/T1): 只有上終板 → anteriorSuperior + [middleSuperior] + posteriorSuperior
        下邊界椎體 (T12/C2): 沒有上終板，不應呼叫此方法
        """
        points = vertebra['points']
        if isinstance(points, dict):
            ant = points.get('anteriorSuperior')
            post = points.get('posteriorSuperior')
            mid = points.get('middleSuperior', None)
            return ant, mid, post
        else:
            if len(points) >= 6:
                # V2.2: [AS, MS, PS, PI, MI, AI]
                return points[0], points[1], points[2]
            else:
                # V2.0: [AS, PS, PI, AI]
                return points[0], None, points[1]

    def calculate_disc_metrics(self, upper_vertebra, lower_vertebra, spine_type='L'):
        """計算椎間盤指標

        椎間盤位於 upper_vertebra 的下終板 與 lower_vertebra 的上終板 之間。
        注意：此處的 upper/lower 指的是解剖學上方/下方（按標註順序排列）。

        V2.2: 使用實際中點距離計算 middleHeight，避免凹陷終板重疊問題
        V2.0/V2.1: middleHeight 為前後平均值（向下相容）
        """
        upper_ant_inf, upper_mid_inf, upper_post_inf = self.get_lower_endplate(upper_vertebra, spine_type)
        lower_ant_sup, lower_mid_sup, lower_post_sup = self.get_upper_endplate(lower_vertebra, spine_type)

        # 椎間盤前方高度
        anterior_height = np.sqrt(
            (lower_ant_sup['x'] - upper_ant_inf['x'])**2 +
            (lower_ant_sup['y'] - upper_ant_inf['y'])**2
        )

        # 椎間盤後方高度
        posterior_height = np.sqrt(
            (lower_post_sup['x'] - upper_post_inf['x'])**2 +
            (lower_post_sup['y'] - upper_post_inf['y'])**2
        )

        # 中間高度: V2.2 使用實際中點距離，否則取平均值
        if upper_mid_inf and lower_mid_sup:
            middle_height = np.sqrt(
                (lower_mid_sup['x'] - upper_mid_inf['x'])**2 +
                (lower_mid_sup['y'] - upper_mid_inf['y'])**2
            )
        else:
            middle_height = (anterior_height + posterior_height) / 2

        # Wedge angle
        upper_angle = np.arctan2(
            upper_post_inf['y'] - upper_ant_inf['y'],
            upper_post_inf['x'] - upper_ant_inf['x']
        )
        lower_angle = np.arctan2(
            lower_post_sup['y'] - lower_ant_sup['y'],
            lower_post_sup['x'] - lower_ant_sup['x']
        )
        wedge_angle = abs(upper_angle - lower_angle) * 180 / np.pi
        if wedge_angle > 90:
            wedge_angle = 180 - wedge_angle

        return {
            'anteriorHeight': float(anterior_height),
            'posteriorHeight': float(posterior_height),
            'middleHeight': float(middle_height),
            'wedgeAngle': float(wedge_angle)
        }

    def prepare_training_data(self, annotations):
        """準備訓練數據"""
        print("📝 準備訓練數據...")

        train_data = []
        val_data = []

        for i, ann in enumerate(tqdm(annotations, desc="處理標註")):
            data = ann['data']
            json_file = ann['file']
            format_version = ann['format']

            # 轉換 V1 格式
            if format_version == 'v1':
                data = self.convert_v1_to_v2(data)
                print(f"  ⚠️ 轉換 V1 格式: {json_file.name}")

            # 自動尋找對應的影像檔案
            image_path = self.find_image_file(json_file)

            spine_type = data.get('spineType', 'L')

            # 計算椎體指標
            vertebrae_with_metrics = []
            for v in data['vertebrae']:
                metrics = self.calculate_vertebra_metrics(v, spine_type)
                vertebrae_with_metrics.append({
                    **v,
                    'metrics': metrics
                })

            # 計算椎間盤指標
            # 椎間盤存在於相鄰的兩個椎體之間
            # 需要排除無法構成椎間盤的邊界組合:
            #   - 下邊界椎體 (T12/C2) 不能作為 disc 的 lower vertebra (它沒有上終板)
            #   - 上邊界椎體 (S1/T1) 不能作為 disc 的 upper vertebra (它沒有下終板)
            discs = []
            for j in range(len(vertebrae_with_metrics) - 1):
                upper = vertebrae_with_metrics[j]
                lower = vertebrae_with_metrics[j + 1]

                upper_bt = upper.get('boundaryType', None)
                lower_bt = lower.get('boundaryType', None)
                upper_boundary = self.is_boundary_vertebra(upper['name'], spine_type, upper_bt, upper.get('points'))
                lower_boundary = self.is_boundary_vertebra(lower['name'], spine_type, lower_bt, lower.get('points'))

                # 上方椎體需要下終板 (完整椎體或下邊界椎體有下終板)
                upper_has_lower_ep = upper_boundary != 'upper'
                # 下方椎體需要上終板 (完整椎體或上邊界椎體有上終板)
                lower_has_upper_ep = lower_boundary != 'lower'

                if upper_has_lower_ep and lower_has_upper_ep:
                    disc_metrics = self.calculate_disc_metrics(upper, lower, spine_type)
                    # 解剖學排序的椎間盤命名
                    disc_name = self.get_disc_level_name(upper['name'], lower['name'], spine_type)
                    discs.append({
                        'level': disc_name,
                        'metrics': disc_metrics
                    })

            # 組裝處理後的數據
            processed_data = {
                'version': '2.0',
                'source_file': str(json_file.name),
                'image_path': image_path,
                'spine_type': data.get('spineType', 'L'),
                'image_info': data.get('imageInfo', {}),
                'vertebrae': vertebrae_with_metrics,
                'discs': discs,
                'abnormalities': self.detect_abnormalities(vertebrae_with_metrics, discs, data.get('spineType', 'L'))
            }

            # 80-20 分割
            if i % 5 == 0:
                val_data.append(processed_data)
            else:
                train_data.append(processed_data)

        # 保存標註檔案
        train_file = self.output_dir / 'annotations' / 'train_annotations.json'
        val_file = self.output_dir / 'annotations' / 'val_annotations.json'

        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 訓練集: {len(train_data)} 個樣本 -> {train_file}")
        print(f"✅ 驗證集: {len(val_data)} 個樣本 -> {val_file}")

        return train_data, val_data

    def find_image_file(self, json_file):
        """尋找對應的影像檔案"""
        json_path = Path(json_file)
        base_name = json_path.stem

        # 嘗試不同的影像格式
        extensions = ['.dcm', '.png', '.jpg', '.jpeg']

        for ext in extensions:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                try:
                    return str(candidate.relative_to(self.input_dir))
                except ValueError:
                    return str(candidate)

        # 嘗試在同目錄下找
        for ext in extensions:
            candidates = list(json_path.parent.glob(f'{base_name}*{ext}'))
            if candidates:
                try:
                    return str(candidates[0].relative_to(self.input_dir))
                except ValueError:
                    return str(candidates[0])

        return ''

    def get_disc_level_name(self, upper_name, lower_name, spine_type):
        """產生解剖學排序的椎間盤名稱

        L-spine: 上方在前 (如 L5/S1, L4/L5)
        C-spine: 上方在前 (如 C3/C4, C6/C7)
        """
        # 定義解剖學順序 (由上到下)
        anatomical_order = {
            'L': ['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'],
            'C': ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1'],
        }
        order = anatomical_order.get(spine_type, [])

        idx_upper = order.index(upper_name) if upper_name in order else -1
        idx_lower = order.index(lower_name) if lower_name in order else -1

        # 確保上方椎體在前
        if idx_upper >= 0 and idx_lower >= 0:
            if idx_upper < idx_lower:
                return f"{upper_name}/{lower_name}"
            else:
                return f"{lower_name}/{upper_name}"

        return f"{upper_name}/{lower_name}"

    def detect_abnormalities(self, vertebrae, discs, spine_type):
        """檢測異常"""
        abnormalities = {
            'compression_fractures': [],
            'listhesis': [],
            'height_progression_issues': []
        }

        # 1. 壓迫性骨折 (只檢查完整椎體)
        for v in vertebrae:
            metrics = v['metrics']
            if metrics.get('anteriorWedging') or metrics.get('compressionFracture'):
                abnormalities['compression_fractures'].append({
                    'vertebra': v['name'],
                    'type': 'anteriorWedging',
                    'ratio': float(metrics['heightRatio']) if metrics['heightRatio'] is not None else 0
                })
            elif metrics.get('crushDeformity'):
                abnormalities['compression_fractures'].append({
                    'vertebra': v['name'],
                    'type': 'crushDeformity',
                    'ratio': float(metrics['heightRatio']) if metrics['heightRatio'] is not None else 0
                })

        # 2. 滑脫檢測（需要至少3個有後緣資訊的椎體）
        # 收集有後緣資訊的椎體 (完整椎體有 posteriorSuperior + posteriorInferior)
        # 上邊界椎體 (S1/T1) 只有 posteriorSuperior → 只能取單點
        # 下邊界椎體 (T12/C2) 只有 posteriorInferior → 只能取單點
        posterior_midpoints = []
        vertebrae_for_listhesis = []
        for v in vertebrae:
            pts = v['points']
            name = v['name']
            bt = v.get('boundaryType', None)
            boundary = self.is_boundary_vertebra(name, spine_type, bt, pts)

            if isinstance(pts, dict):
                if boundary == 'upper':
                    # 上邊界 (S1/T1): 只有上終板 → posteriorSuperior
                    mid_x = pts['posteriorSuperior']['x']
                    mid_y = pts['posteriorSuperior']['y']
                elif boundary == 'lower':
                    # 下邊界 (T12/C2): 只有下終板 → posteriorInferior
                    mid_x = pts['posteriorInferior']['x']
                    mid_y = pts['posteriorInferior']['y']
                else:
                    # 完整椎體: 後緣中點
                    mid_x = (pts['posteriorSuperior']['x'] + pts['posteriorInferior']['x']) / 2
                    mid_y = (pts['posteriorSuperior']['y'] + pts['posteriorInferior']['y']) / 2
            else:
                if len(pts) >= 6:
                    # V2.2: [AS, MS, PS, PI, MI, AI] → posterior = pts[2], pts[3]
                    mid_x = (pts[2]['x'] + pts[3]['x']) / 2
                    mid_y = (pts[2]['y'] + pts[3]['y']) / 2
                elif len(pts) >= 4:
                    # V2.0: [AS, PS, PI, AI] → posterior = pts[1], pts[2]
                    mid_x = (pts[1]['x'] + pts[2]['x']) / 2
                    mid_y = (pts[1]['y'] + pts[2]['y']) / 2
                elif len(pts) >= 2:
                    mid_x = pts[1]['x'] if len(pts) > 1 else pts[0]['x']
                    mid_y = pts[1]['y'] if len(pts) > 1 else pts[0]['y']
                else:
                    continue

            posterior_midpoints.append({'name': name, 'x': mid_x, 'y': mid_y})
            vertebrae_for_listhesis.append(v)

        if len(posterior_midpoints) >= 3:
            first = posterior_midpoints[0]
            last = posterior_midpoints[-1]

            for i in range(1, len(posterior_midpoints) - 1):
                p = posterior_midpoints[i]

                line_len = np.sqrt((last['x'] - first['x'])**2 + (last['y'] - first['y'])**2)
                if line_len > 0:
                    distance = abs(
                        (last['y'] - first['y']) * p['x'] -
                        (last['x'] - first['x']) * p['y'] +
                        last['x'] * first['y'] - last['y'] * first['x']
                    ) / line_len

                    # 計算椎體寬度
                    v = vertebrae_for_listhesis[i]
                    pts = v['points']
                    if isinstance(pts, dict):
                        if 'anteriorSuperior' in pts and 'posteriorSuperior' in pts:
                            width = abs(pts['posteriorSuperior']['x'] - pts['anteriorSuperior']['x'])
                        elif 'anteriorInferior' in pts and 'posteriorInferior' in pts:
                            width = abs(pts['posteriorInferior']['x'] - pts['anteriorInferior']['x'])
                        else:
                            width = 100  # fallback
                    else:
                        width = abs(pts[1]['x'] - pts[0]['x']) if len(pts) >= 2 else 100

                    shift_percent = (distance / width) * 100 if width > 0 else 0

                    if shift_percent > 5:
                        expected_x = first['x'] + (p['y'] - first['y']) * (last['x'] - first['x']) / (last['y'] - first['y']) if (last['y'] - first['y']) != 0 else first['x']
                        listhesis_type = 'retrolisthesis' if p['x'] > expected_x else 'anterolisthesis'

                        abnormalities['listhesis'].append({
                            'vertebra': p['name'],
                            'type': listhesis_type,
                            'shift_percent': float(shift_percent)
                        })

        # 3. 椎間盤高度遞進檢查
        if len(discs) >= 2:
            heights = [d['metrics']['middleHeight'] for d in discs]

            if spine_type == 'L':
                # L-spine: L4/5 應該最高
                l45_idx = next((i for i, d in enumerate(discs) if 'L4/L5' in d['level']), None)

                if l45_idx is not None:
                    l45_height = heights[l45_idx]
                    for i, d in enumerate(discs):
                        if i != l45_idx and 'L5/S1' not in d['level']:
                            if heights[i] > l45_height * 1.1:
                                abnormalities['height_progression_issues'].append({
                                    'level': d['level'],
                                    'issue': 'height_exceeds_L4L5'
                                })

            elif spine_type == 'C':
                # C-spine: 應該越來越高
                for i in range(len(heights) - 1):
                    if heights[i] > heights[i + 1] * 1.2:
                        abnormalities['height_progression_issues'].append({
                            'level': discs[i]['level'],
                            'issue': 'height_not_increasing'
                        })

        return abnormalities

    def analyze_dataset(self, train_data, val_data):
        """分析數據集統計資訊"""
        print("\n📊 數據集分析:")

        all_data = train_data + val_data

        print(f"  總樣本數: {len(all_data)}")
        print(f"  訓練樣本: {len(train_data)}")
        print(f"  驗證樣本: {len(val_data)}")

        # 脊椎類型統計
        spine_types = {}
        for d in all_data:
            st = d.get('spine_type', 'L')
            spine_types[st] = spine_types.get(st, 0) + 1
        print(f"\n  脊椎類型分布: {spine_types}")

        # 椎體統計
        total_vertebrae = sum(len(d['vertebrae']) for d in all_data)
        avg_vertebrae = total_vertebrae / len(all_data) if all_data else 0
        print(f"\n  總椎體數: {total_vertebrae}")
        print(f"  平均每張影像: {avg_vertebrae:.1f} 個椎體")

        # 椎體名稱分布
        vertebra_counts = {}
        for d in all_data:
            for v in d['vertebrae']:
                name = v['name']
                vertebra_counts[name] = vertebra_counts.get(name, 0) + 1

        print(f"\n  椎體分布:")
        for name in sorted(vertebra_counts.keys()):
            print(f"    {name}: {vertebra_counts[name]}")

        # 異常統計
        total_fractures = sum(len(d['abnormalities']['compression_fractures']) for d in all_data)
        total_listhesis = sum(len(d['abnormalities']['listhesis']) for d in all_data)
        total_height_issues = sum(len(d['abnormalities']['height_progression_issues']) for d in all_data)

        print(f"\n  異常統計:")
        print(f"    壓迫性骨折: {total_fractures}")
        print(f"    滑脫: {total_listhesis}")
        print(f"    高度遞進異常: {total_height_issues}")

    def create_dataset_info(self):
        """創建數據集資訊檔案"""
        info = {
            "dataset_name": "Spine Vertebra Corner Detection Dataset V2.1",
            "description": "脊椎椎體頂點檢測機器學習數據集 - 完整椎體4角點, 邊界椎體2點",
            "version": "2.1",
            "annotation_format": {
                "vertebrae": [
                    {
                        "name": "椎體名稱 (如 L4)",
                        "points": {
                            "anteriorSuperior": "前上角座標 {x, y}",
                            "posteriorSuperior": "後上角座標 {x, y}",
                            "posteriorInferior": "後下角座標 {x, y}",
                            "anteriorInferior": "前下角座標 {x, y}"
                        },
                        "metrics": {
                            "anteriorHeight": "前緣高度 (px)",
                            "posteriorHeight": "後緣高度 (px)",
                            "heightRatio": "前/後比例",
                            "compressionFracture": "是否壓迫性骨折 (前緣 < 後緣 * 0.75)"
                        }
                    }
                ],
                "discs": [
                    {
                        "level": "椎間盤標籤 (如 L4/L5)",
                        "metrics": {
                            "anteriorHeight": "前方高度",
                            "posteriorHeight": "後方高度",
                            "middleHeight": "平均高度",
                            "wedgeAngle": "楔形角度"
                        }
                    }
                ],
                "abnormalities": {
                    "compression_fractures": "壓迫性骨折列表",
                    "listhesis": "滑脫列表 (>5% 後緣偏移)",
                    "height_progression_issues": "高度遞進異常"
                }
            },
            "model_tasks": [
                "vertebra_corner_detection",
                "keypoint_regression"
            ],
            "clinical_rules": {
                "compression_fracture": "anterior_height < posterior_height * 0.75",
                "listhesis_threshold": "5% vertebra width",
                "L_spine_height_pattern": "L4/L5 should be highest, L5/S1 can be smaller",
                "C_spine_height_pattern": "heights should increase caudally"
            }
        }

        with open(self.output_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print("\n✅ 創建數據集資訊檔案")

    def process_all(self, spine_type_filter=None):
        """處理所有數據

        Args:
            spine_type_filter: 'L', 'C', or None (all)
        """
        filter_msg = f" (filter: {spine_type_filter}-spine only)" if spine_type_filter else ""
        print(f"開始椎體頂點檢測數據準備流程 V2{filter_msg}...")

        # 1. 收集標註檔案
        annotations = self.collect_annotations()

        # 過濾脊椎類型
        if spine_type_filter:
            before = len(annotations)
            annotations = [
                a for a in annotations
                if a['data'].get('spineType', a['data'].get('spine_type', 'L')) == spine_type_filter
            ]
            print(f"  Filtered: {before} -> {len(annotations)} ({spine_type_filter}-spine)")

        if not annotations:
            print("❌ 未找到有效的標註檔案")
            print("💡 請使用 spinal-annotation-web.html 進行標註")
            return

        # 2. 準備訓練數據
        train_data, val_data = self.prepare_training_data(annotations)

        # 3. 分析數據集
        self.analyze_dataset(train_data, val_data)

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
    parser = argparse.ArgumentParser(description='椎體頂點檢測數據準備 V2')
    parser.add_argument('--input_dir', type=str, default='.',
                       help='標註檔案目錄（預設為當前目錄）')
    parser.add_argument('--output_dir', type=str, default='endplate_training_data',
                       help='輸出目錄')
    parser.add_argument('--spine-type', type=str, default=None, choices=['L', 'C'],
                       help='只處理特定脊椎類型 (L=lumbar, C=cervical)')

    args = parser.parse_args()

    preparer = VertebraDataPreparer(args.input_dir, args.output_dir)
    preparer.process_all(spine_type_filter=args.spine_type)

if __name__ == "__main__":
    main()
