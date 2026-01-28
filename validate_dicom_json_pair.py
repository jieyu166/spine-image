#!/usr/bin/env python3
"""
DICOM與JSON標註檔案配對驗證工具
DICOM-JSON Pair Validation Tool

功能：
- 驗證DICOM檔案與JSON標註檔案的配對關係
- 檢查檔案完整性、時間戳記、患者資訊等
- 提供詳細的驗證報告和建議

作者: AI Assistant
日期: 2024
"""

import os
import json
import hashlib
import datetime
from pathlib import Path
import pydicom
from pydicom.errors import InvalidDicomError
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DicomJsonValidator:
    """DICOM與JSON檔案配對驗證器"""
    
    def __init__(self):
        self.validation_results = {
            'file_exists': False,
            'dicom_valid': False,
            'json_valid': False,
            'patient_match': False,
            'study_match': False,
            'date_match': False,
            'image_dimensions_match': False,
            'file_integrity': False,
            'timestamp_consistency': False,
            'overall_match': False,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
    
    def validate_pair(self, dicom_path, json_path):
        """驗證DICOM與JSON檔案配對"""
        logger.info(f"開始驗證檔案配對: {dicom_path} <-> {json_path}")
        
        # 重置結果
        self.validation_results = {key: False if key not in ['issues', 'warnings', 'recommendations'] 
                                 else [] for key in self.validation_results.keys()}
        
        # 1. 檢查檔案存在性
        self._check_file_existence(dicom_path, json_path)
        
        if not self.validation_results['file_exists']:
            return self.validation_results
        
        # 2. 驗證DICOM檔案
        dicom_data = self._validate_dicom_file(dicom_path)
        
        # 3. 驗證JSON檔案
        json_data = self._validate_json_file(json_path)
        
        if not (self.validation_results['dicom_valid'] and self.validation_results['json_valid']):
            return self.validation_results
        
        # 4. 比較患者資訊
        self._compare_patient_info(dicom_data, json_data)
        
        # 5. 比較檢查資訊
        self._compare_study_info(dicom_data, json_data)
        
        # 6. 比較日期資訊
        self._compare_dates(dicom_data, json_data)
        
        # 7. 比較影像尺寸
        self._compare_image_dimensions(dicom_data, json_data)
        
        # 8. 檢查檔案完整性
        self._check_file_integrity(dicom_path, json_path)
        
        # 9. 檢查時間戳記一致性
        self._check_timestamp_consistency(dicom_path, json_path)
        
        # 10. 計算整體匹配度
        self._calculate_overall_match()
        
        return self.validation_results
    
    def _check_file_existence(self, dicom_path, json_path):
        """檢查檔案是否存在"""
        dicom_exists = os.path.exists(dicom_path)
        json_exists = os.path.exists(json_path)
        
        if not dicom_exists:
            self.validation_results['issues'].append(f"DICOM檔案不存在: {dicom_path}")
        
        if not json_exists:
            self.validation_results['issues'].append(f"JSON檔案不存在: {json_path}")
        
        self.validation_results['file_exists'] = dicom_exists and json_exists
        
        if self.validation_results['file_exists']:
            logger.info("✓ 檔案存在性檢查通過")
        else:
            logger.error("✗ 檔案存在性檢查失敗")
    
    def _validate_dicom_file(self, dicom_path):
        """驗證DICOM檔案"""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            self.validation_results['dicom_valid'] = True
            logger.info("✓ DICOM檔案格式有效")
            return dicom_data
        except InvalidDicomError as e:
            self.validation_results['issues'].append(f"DICOM檔案格式無效: {e}")
            logger.error(f"✗ DICOM檔案格式無效: {e}")
            return None
        except Exception as e:
            self.validation_results['issues'].append(f"讀取DICOM檔案時發生錯誤: {e}")
            logger.error(f"✗ 讀取DICOM檔案時發生錯誤: {e}")
            return None
    
    def _validate_json_file(self, json_path):
        """驗證JSON檔案"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 檢查必要的欄位
            required_fields = ['metadata', 'measurements']
            for field in required_fields:
                if field not in json_data:
                    self.validation_results['issues'].append(f"JSON檔案缺少必要欄位: {field}")
                    return None
            
            self.validation_results['json_valid'] = True
            logger.info("✓ JSON檔案格式有效")
            return json_data
        except json.JSONDecodeError as e:
            self.validation_results['issues'].append(f"JSON檔案格式無效: {e}")
            logger.error(f"✗ JSON檔案格式無效: {e}")
            return None
        except Exception as e:
            self.validation_results['issues'].append(f"讀取JSON檔案時發生錯誤: {e}")
            logger.error(f"✗ 讀取JSON檔案時發生錯誤: {e}")
            return None
    
    def _compare_patient_info(self, dicom_data, json_data):
        """比較患者資訊"""
        if not dicom_data or not json_data:
            return
        
        # 提取DICOM患者資訊
        dicom_patient_id = getattr(dicom_data, 'PatientID', '')
        dicom_patient_name = getattr(dicom_data, 'PatientName', '')
        
        # 提取JSON患者資訊
        json_patient_id = json_data.get('metadata', {}).get('patient_id', '')
        
        # 比較患者ID
        if dicom_patient_id and json_patient_id:
            if dicom_patient_id == json_patient_id:
                self.validation_results['patient_match'] = True
                logger.info("✓ 患者ID匹配")
            else:
                self.validation_results['issues'].append(
                    f"患者ID不匹配: DICOM={dicom_patient_id}, JSON={json_patient_id}"
                )
                logger.warning(f"⚠ 患者ID不匹配: DICOM={dicom_patient_id}, JSON={json_patient_id}")
        else:
            self.validation_results['warnings'].append("無法比較患者ID（缺少資訊）")
    
    def _compare_study_info(self, dicom_data, json_data):
        """比較檢查資訊"""
        if not dicom_data or not json_data:
            return
        
        # 提取DICOM檢查資訊
        dicom_study_uid = getattr(dicom_data, 'StudyInstanceUID', '')
        dicom_study_id = getattr(dicom_data, 'StudyID', '')
        
        # 提取JSON檢查資訊
        json_study_id = json_data.get('metadata', {}).get('study_id', '')
        
        # 比較檢查ID
        if dicom_study_id and json_study_id:
            if dicom_study_id == json_study_id:
                self.validation_results['study_match'] = True
                logger.info("✓ 檢查ID匹配")
            else:
                self.validation_results['issues'].append(
                    f"檢查ID不匹配: DICOM={dicom_study_id}, JSON={json_study_id}"
                )
                logger.warning(f"⚠ 檢查ID不匹配: DICOM={dicom_study_id}, JSON={json_study_id}")
        else:
            self.validation_results['warnings'].append("無法比較檢查ID（缺少資訊）")
    
    def _compare_dates(self, dicom_data, json_data):
        """比較日期資訊"""
        if not dicom_data or not json_data:
            return
        
        # 提取DICOM日期
        dicom_study_date = getattr(dicom_data, 'StudyDate', '')
        dicom_study_time = getattr(dicom_data, 'StudyTime', '')
        
        # 提取JSON日期
        json_study_date = json_data.get('metadata', {}).get('study_date', '')
        
        # 比較日期
        if dicom_study_date and json_study_date:
            # 正規化日期格式
            dicom_date = self._normalize_date(dicom_study_date)
            json_date = self._normalize_date(json_study_date)
            
            if dicom_date == json_date:
                self.validation_results['date_match'] = True
                logger.info("✓ 檢查日期匹配")
            else:
                self.validation_results['issues'].append(
                    f"檢查日期不匹配: DICOM={dicom_date}, JSON={json_date}"
                )
                logger.warning(f"⚠ 檢查日期不匹配: DICOM={dicom_date}, JSON={json_date}")
        else:
            self.validation_results['warnings'].append("無法比較檢查日期（缺少資訊）")
    
    def _compare_image_dimensions(self, dicom_data, json_data):
        """比較影像尺寸"""
        if not dicom_data or not json_data:
            return
        
        # 提取DICOM影像尺寸
        dicom_rows = getattr(dicom_data, 'Rows', 0)
        dicom_cols = getattr(dicom_data, 'Columns', 0)
        
        # 提取JSON影像尺寸
        json_dims = json_data.get('metadata', {}).get('image_dimensions', {})
        json_width = json_dims.get('width', 0)
        json_height = json_dims.get('height', 0)
        
        # 比較尺寸
        if dicom_rows and dicom_cols and json_width and json_height:
            if dicom_rows == json_height and dicom_cols == json_width:
                self.validation_results['image_dimensions_match'] = True
                logger.info("✓ 影像尺寸匹配")
            else:
                self.validation_results['issues'].append(
                    f"影像尺寸不匹配: DICOM={dicom_cols}x{dicom_rows}, JSON={json_width}x{json_height}"
                )
                logger.warning(f"⚠ 影像尺寸不匹配: DICOM={dicom_cols}x{dicom_rows}, JSON={json_width}x{json_height}")
        else:
            self.validation_results['warnings'].append("無法比較影像尺寸（缺少資訊）")
    
    def _check_file_integrity(self, dicom_path, json_path):
        """檢查檔案完整性"""
        try:
            # 計算檔案雜湊值
            dicom_hash = self._calculate_file_hash(dicom_path)
            json_hash = self._calculate_file_hash(json_path)
            
            if dicom_hash and json_hash:
                self.validation_results['file_integrity'] = True
                logger.info("✓ 檔案完整性檢查通過")
            else:
                self.validation_results['issues'].append("檔案完整性檢查失敗")
        except Exception as e:
            self.validation_results['issues'].append(f"檔案完整性檢查時發生錯誤: {e}")
    
    def _check_timestamp_consistency(self, dicom_path, json_path):
        """檢查時間戳記一致性"""
        try:
            # 獲取檔案修改時間
            dicom_mtime = os.path.getmtime(dicom_path)
            json_mtime = os.path.getmtime(json_path)
            
            # 檢查時間差（允許5分鐘的誤差）
            time_diff = abs(dicom_mtime - json_mtime)
            if time_diff <= 300:  # 5分鐘
                self.validation_results['timestamp_consistency'] = True
                logger.info("✓ 時間戳記一致性檢查通過")
            else:
                self.validation_results['warnings'].append(
                    f"檔案修改時間差異較大: {time_diff/60:.1f}分鐘"
                )
                logger.warning(f"⚠ 檔案修改時間差異較大: {time_diff/60:.1f}分鐘")
        except Exception as e:
            self.validation_results['issues'].append(f"時間戳記檢查時發生錯誤: {e}")
    
    def _calculate_overall_match(self):
        """計算整體匹配度"""
        checks = [
            'file_exists', 'dicom_valid', 'json_valid', 'patient_match',
            'study_match', 'date_match', 'image_dimensions_match',
            'file_integrity', 'timestamp_consistency'
        ]
        
        passed_checks = sum(1 for check in checks if self.validation_results[check])
        total_checks = len(checks)
        
        match_percentage = (passed_checks / total_checks) * 100
        self.validation_results['overall_match'] = match_percentage >= 80
        
        # 生成建議
        self._generate_recommendations(match_percentage)
    
    def _generate_recommendations(self, match_percentage):
        """生成建議"""
        if match_percentage >= 90:
            self.validation_results['recommendations'].append("檔案配對品質優秀，可以直接使用")
        elif match_percentage >= 80:
            self.validation_results['recommendations'].append("檔案配對品質良好，建議檢查警告項目")
        elif match_percentage >= 60:
            self.validation_results['recommendations'].append("檔案配對品質一般，需要修正部分問題")
        else:
            self.validation_results['recommendations'].append("檔案配對品質較差，建議重新檢查檔案來源")
        
        # 針對特定問題的建議
        if not self.validation_results['patient_match']:
            self.validation_results['recommendations'].append("建議檢查患者ID是否正確")
        
        if not self.validation_results['date_match']:
            self.validation_results['recommendations'].append("建議檢查檢查日期是否正確")
        
        if not self.validation_results['image_dimensions_match']:
            self.validation_results['recommendations'].append("建議檢查影像尺寸是否正確")
    
    def _normalize_date(self, date_str):
        """正規化日期格式"""
        if not date_str:
            return ""
        
        # 處理YYYYMMDD格式
        if len(date_str) == 8 and date_str.isdigit():
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # 處理YYYY-MM-DD格式
        if len(date_str) == 10 and date_str.count('-') == 2:
            return date_str
        
        return date_str
    
    def _calculate_file_hash(self, file_path):
        """計算檔案雜湊值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None
    
    def generate_report(self, output_path=None):
        """生成驗證報告"""
        report = {
            'validation_timestamp': datetime.datetime.now().isoformat(),
            'summary': {
                'overall_match': self.validation_results['overall_match'],
                'total_checks': 9,
                'passed_checks': sum(1 for key in ['file_exists', 'dicom_valid', 'json_valid', 
                                                  'patient_match', 'study_match', 'date_match',
                                                  'image_dimensions_match', 'file_integrity', 
                                                  'timestamp_consistency'] 
                                    if self.validation_results[key]),
                'issues_count': len(self.validation_results['issues']),
                'warnings_count': len(self.validation_results['warnings'])
            },
            'detailed_results': self.validation_results
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"驗證報告已儲存至: {output_path}")
        
        return report

def validate_dicom_json_pair(dicom_path, json_path, output_report=None):
    """驗證DICOM與JSON檔案配對的主函數"""
    validator = DicomJsonValidator()
    results = validator.validate_pair(dicom_path, json_path)
    
    # 生成報告
    if output_report:
        validator.generate_report(output_report)
    
    return results

def batch_validate_pairs(directory_path, output_dir=None):
    """批量驗證目錄中的DICOM-JSON配對"""
    directory = Path(directory_path)
    results = []
    
    # 尋找所有DICOM檔案
    dicom_files = list(directory.glob('**/*.dcm')) + list(directory.glob('**/*.DCM'))
    
    for dicom_file in dicom_files:
        # 尋找對應的JSON檔案
        json_file = dicom_file.with_suffix('.json')
        if not json_file.exists():
            # 嘗試其他命名模式
            json_file = dicom_file.parent / f"{dicom_file.stem}_annotations.json"
            if not json_file.exists():
                json_file = dicom_file.parent / f"{dicom_file.stem}_measurements.json"
        
        if json_file.exists():
            logger.info(f"驗證配對: {dicom_file.name} <-> {json_file.name}")
            result = validate_dicom_json_pair(str(dicom_file), str(json_file))
            results.append({
                'dicom_file': str(dicom_file),
                'json_file': str(json_file),
                'validation_result': result
            })
        else:
            logger.warning(f"找不到對應的JSON檔案: {dicom_file}")
    
    # 生成批量報告
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        batch_report = {
            'batch_validation_timestamp': datetime.datetime.now().isoformat(),
            'total_pairs': len(results),
            'valid_pairs': sum(1 for r in results if r['validation_result']['overall_match']),
            'results': results
        }
        
        report_path = output_dir / 'batch_validation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(batch_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"批量驗證報告已儲存至: {report_path}")
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python validate_dicom_json_pair.py <dicom_file> <json_file> [output_report]")
        print("  python validate_dicom_json_pair.py --batch <directory> [output_dir]")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        # 批量驗證模式
        directory = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        results = batch_validate_pairs(directory, output_dir)
        print(f"批量驗證完成，共處理 {len(results)} 個配對")
    else:
        # 單一檔案驗證模式
        dicom_file = sys.argv[1]
        json_file = sys.argv[2]
        output_report = sys.argv[3] if len(sys.argv) > 3 else None
        
        results = validate_dicom_json_pair(dicom_file, json_file, output_report)
        
        print("\n=== 驗證結果 ===")
        print(f"整體匹配: {'✓' if results['overall_match'] else '✗'}")
        print(f"問題數量: {len(results['issues'])}")
        print(f"警告數量: {len(results['warnings'])}")
        
        if results['issues']:
            print("\n問題:")
            for issue in results['issues']:
                print(f"  - {issue}")
        
        if results['warnings']:
            print("\n警告:")
            for warning in results['warnings']:
                print(f"  - {warning}")
        
        if results['recommendations']:
            print("\n建議:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
