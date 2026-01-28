#!/usr/bin/env python3
"""
3D Slicer 脊椎標註匯出腳本
Spine Annotation Export Script for 3D Slicer

功能：
- 匯出脊椎角度測量標註為JSON格式
- 自動計算椎體前後緣
- 支援角度銳角化處理
- 建立L1-S1範本線
- 包含完整的臨床資訊

作者: AI Assistant
日期: 2024
"""

import json
import math
import re
import datetime
import slicer
import qt
import vtk

# ===== 工具函數 =====

def _activeVolumeNode():
    """獲取當前活動的體積節點"""
    lm = slicer.app.layoutManager()
    if not lm:
        return None
    red = lm.sliceWidget('Red').mrmlSliceCompositeNode()
    vid = red.GetBackgroundVolumeID()
    if not vid:
        return None
    return slicer.util.getNode(vid)

def _getImageInfo(volumeNode):
    """獲取影像資訊"""
    img = volumeNode.GetImageData()
    dims = img.GetDimensions() if img else (0, 0, 0)
    storage = volumeNode.GetStorageNode()
    image_path = storage.GetFileName() if storage and storage.GetFileName() else ""
    return {
        "image_path": image_path,
        "image_dimensions": {"width": int(dims[0]), "height": int(dims[1])}
    }

def _dicomMetaFromNode(volumeNode):
    """從DICOM節點提取元數據"""
    meta = {"patient_id": "", "study_id": "", "study_date": ""}
    pid = volumeNode.GetAttribute('DICOM.PatientID')
    suid = volumeNode.GetAttribute('DICOM.StudyInstanceUID')
    sdate = volumeNode.GetAttribute('DICOM.StudyDate')
    if pid:
        meta["patient_id"] = pid
    if suid:
        meta["study_id"] = suid
    if sdate and len(sdate) == 8:
        meta["study_date"] = f"{sdate[0:4]}-{sdate[4:6]}-{sdate[6:8]}"
    return meta

def _rasToIjk(volumeNode, rasPoint):
    """將RAS座標轉換為IJK座標"""
    rasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjk)
    p = list(rasPoint) + [1.0]
    ijk = [0.0, 0.0, 0.0, 0.0]
    rasToIjk.MultiplyPoint(p, ijk)
    return (ijk[0], ijk[1], ijk[2])

# 正則表達式用於解析椎間隙標籤
_LEVEL_RE = re.compile(r"([CTLS]\d+)[_/]([CTLS]\d+)", re.I)

def _normalizeLevel(name):
    """正規化椎間隙標籤"""
    m = _LEVEL_RE.search(name.replace(" ", ""))
    if not m:
        return None
    return f"{m.group(1).upper()}/{m.group(2).upper()}"

def _lineTypeFromName(name):
    """從節點名稱判斷線條類型"""
    n = name.lower()
    if "lower" in n:
        return "lowerEndplate"
    if "upper" in n:
        return "upperEndplate"
    return None

def _angleDegrees(p1, p2):
    """計算兩點間的角度（度）"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

def _normalize_angle_to_acute(diff_deg: float) -> float:
    """將角度正規化為銳角（≤90°）"""
    diff = abs(diff_deg) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff

# ===== 收集標註 =====

def _gatherLines():
    """收集所有標註線條"""
    groups = {}
    for node in slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode'):
        lvl = _normalizeLevel(node.GetName() or "")
        if not lvl:
            continue
        t = _lineTypeFromName(node.GetName() or "")
        if not t:
            continue
        if node.GetNumberOfControlPoints() < 2:
            continue
        vol = _activeVolumeNode()
        if not vol:
            continue
        
        # 獲取世界座標
        ras0, ras1 = [0, 0, 0], [0, 0, 0]
        node.GetNthControlPointPositionWorld(0, ras0)
        node.GetNthControlPointPositionWorld(1, ras1)
        
        # 轉換為IJK座標
        ijk0, ijk1 = _rasToIjk(vol, ras0), _rasToIjk(vol, ras1)
        p0 = {"x": float(ijk0[0]), "y": float(ijk0[1])}
        p1 = {"x": float(ijk1[0]), "y": float(ijk1[1])}
        
        # 獲取信心度和註記
        conf = node.GetAttribute("confidence")
        confidence = float(conf) if conf else 0.95  # 預設信心度
        notes = node.GetAttribute("notes") or ""
        
        if lvl not in groups:
            groups[lvl] = {}
        groups[lvl][t] = (p0, p1, confidence, notes)
    
    return groups

# ===== 計算椎體前後緣 =====

def _build_vertebra_edges(measurements):
    """建立椎體前後緣資料"""
    level_dict = {m["level"]: m for m in measurements}
    vertebra_edges = {}
    
    for lvl, m in level_dict.items():
        sup, inf = lvl.split("/")
        sup, inf = sup.upper(), inf.upper()
        
        # 上界：這層的upperEndplate
        if "upperEndplate" not in m:
            continue
        upper_line = m["upperEndplate"]
        upper_ant = min(upper_line, key=lambda p: p["x"])  # x較小為前緣
        upper_post = max(upper_line, key=lambda p: p["x"])  # x較大為後緣
        
        # 下界：尋找與inf有關的lowerEndplate
        lower_line = None
        for k, v in level_dict.items():
            if k.startswith(inf + "/") and "lowerEndplate" in v:
                lower_line = v["lowerEndplate"]
                break
        
        if not lower_line:
            continue
        
        lower_ant = min(lower_line, key=lambda p: p["x"])
        lower_post = max(lower_line, key=lambda p: p["x"])
        
        # 每個椎體用「上一層lowerEndplate + 下一層upperEndplate」組成
        vertebra_edges[inf] = {
            "anterior": [upper_ant, lower_ant],
            "posterior": [upper_post, lower_post]
        }
    
    return vertebra_edges

# ===== 主程式 =====

def export_spine_annotations():
    """匯出脊椎標註的主函數"""
    vol = _activeVolumeNode()
    if not vol:
        slicer.util.errorDisplay("No background volume found.")
        return
    
    groups = _gatherLines()
    if not groups:
        slicer.util.errorDisplay("No MarkupsLine nodes found.")
        return
    
    dicomMeta = _dicomMetaFromNode(vol)
    imageInfo = _getImageInfo(vol)
    
    # 建立UI對話框
    d = qt.QDialog(slicer.util.mainWindow())
    d.setWindowTitle("Spine Annotation Export")
    d.setMinimumWidth(500)
    d.setMinimumHeight(400)
    
    mainLayout = qt.QVBoxLayout(d)
    formLayout = qt.QFormLayout()
    
    # 脊椎類型選擇
    spineType = qt.QComboBox()
    spineType.addItems(["L", "C", "T", "S"])
    spineType.setCurrentText("L")  # 預設腰椎
    
    # 影像類型選擇
    imageType = qt.QComboBox()
    imageType.addItems(["flexion", "extension", "neutral"])
    imageType.setCurrentText("flexion")  # 預設屈曲位
    
    # 標註者資訊
    annotName = qt.QLineEdit()
    annotName.setPlaceholderText("Enter annotator name")
    
    annotId = qt.QLineEdit()
    annotId.setPlaceholderText("Enter annotator ID")
    
    annotSpec = qt.QLineEdit()
    annotSpec.setPlaceholderText("e.g., Radiology")
    annotSpec.setText("Radiology")  # 預設專科
    
    annotYrs = qt.QLineEdit()
    annotYrs.setPlaceholderText("Years of experience")
    
    # 原始報告文字框
    report = qt.QPlainTextEdit()
    report.setPlaceholderText("Paste original clinical report here...")
    report.setFixedHeight(60)  # 預設60px高
    report.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
    
    # 添加表單項目
    formLayout.addRow("Spine type:", spineType)
    formLayout.addRow("Image type:", imageType)
    formLayout.addRow("Annotator name:", annotName)
    formLayout.addRow("Annotator ID:", annotId)
    formLayout.addRow("Specialty:", annotSpec)
    formLayout.addRow("Experience years:", annotYrs)
    formLayout.addRow("Original report:", report)
    
    mainLayout.addLayout(formLayout)
    
    # 按鈕區域
    okButton = qt.QPushButton("OK")
    cancelButton = qt.QPushButton("Cancel")
    
    okButton.clicked.connect(lambda checked=False: d.accept())
    cancelButton.clicked.connect(lambda checked=False: d.reject())
    
    buttonLayout = qt.QHBoxLayout()
    buttonLayout.addStretch(1)
    buttonLayout.addWidget(okButton)
    buttonLayout.addWidget(cancelButton)
    
    mainLayout.addLayout(buttonLayout)
    d.setLayout(mainLayout)
    
    # 顯示對話框
    if d.exec_() != qt.QDialog.Accepted:
        return
    
    # 建立輸出資料結構
    out = {
        "metadata": {
            "patient_id": dicomMeta["patient_id"],
            "study_id": dicomMeta["study_id"],
            "study_date": dicomMeta["study_date"],
            "spine_type": spineType.currentText,
            "image_type": imageType.currentText,
            "image_path": imageInfo["image_path"],
            "image_dimensions": imageInfo["image_dimensions"],
            "annotator": {
                "name": annotName.text.strip(),
                "id": annotId.text.strip(),
                "specialty": annotSpec.text.strip(),
                "experience_years": int(annotYrs.text) if annotYrs.text.strip().isdigit() else None
            },
            "annotation_date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "annotation_version": "1.0"
        },
        "measurements": [],
        "clinical_notes": {
            "original_report": report.toPlainText()
        }
    }
    
    # 處理每個椎間隙的測量
    for lvl, parts in groups.items():
        if "lowerEndplate" not in parts or "upperEndplate" not in parts:
            continue
        
        l0, l1, confL, notesL = parts["lowerEndplate"]
        u0, u1, confU, notesU = parts["upperEndplate"]
        
        # 計算角度
        a1 = _angleDegrees((l0["x"], l0["y"]), (l1["x"], l1["y"]))
        a2 = _angleDegrees((u0["x"], u0["y"]), (u1["x"], u1["y"]))
        raw = abs(a1 - a2) % 180.0
        angle = _normalize_angle_to_acute(raw)  # 強制銳角化
        
        # 建立測量項目
        entry = {
            "level": lvl,
            "angle": round(angle, 1),
            "angle_raw": round(raw, 1),  # 保留原始角度
            "confidence": round((confL + confU) / 2, 3),
            "measurement_method": "manual",
            "lowerEndplate": [l0, l1],
            "upperEndplate": [u0, u1]
        }
        
        # 添加註記
        if notesL or notesU:
            entry["measurement_notes"] = "; ".join([notesL, notesU])
        
        out["measurements"].append(entry)
    
    # 加入椎體前後緣資料
    out["vertebra_edges"] = _build_vertebra_edges(out["measurements"])
    
    # 添加整體評估
    if out["measurements"]:
        angles = [m["angle"] for m in out["measurements"]]
        out["overall_assessment"] = {
            "total_vertebrae_measured": len(out["measurements"]),
            "average_angle": round(sum(angles) / len(angles), 1),
            "angle_range": f"{min(angles)}-{max(angles)}",
            "stability_assessment": "stable" if max(angles) - min(angles) < 10 else "variable"
        }
    
    # 儲存檔案
    path = qt.QFileDialog.getSaveFileName(
        slicer.util.mainWindow(),
        "Save Spine Annotations",
        "",
        "JSON Files (*.json)"
    )
    
    if not path:
        return
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    
    slicer.util.infoDisplay(
        f"Successfully exported {len(out['measurements'])} levels to:\n{path}\n\n"
        f"Added {len(out.get('vertebra_edges', {}))} vertebra edge definitions."
    )

# ===== 範本線建立 =====

def create_spine_template_lines():
    """建立L1-S1範本線"""
    pairs = [("L1", "L2"), ("L2", "L3"), ("L3", "L4"), ("L4", "L5"), ("L5", "S1")]
    
    created_count = 0
    for a, b in pairs:
        for kind in ("lower", "upper"):
            name = f"{a}_{b}_{kind}Endplate"
            
            # 檢查是否已存在
            if slicer.util.getNode(name) is not None:
                continue
            
            # 建立新的線條節點
            ln = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
            ln.AddControlPoint(0, 0, 0)
            ln.AddControlPoint(5, 0, 0)
            
            # 設定預設屬性
            ln.SetAttribute("confidence", "0.95")  # 預設信心度
            ln.SetAttribute("notes", "")
            
            created_count += 1
    
    if created_count > 0:
        slicer.util.infoDisplay(f"Created {created_count} template lines for L1–S1.")
    else:
        slicer.util.infoDisplay("All template lines already exist.")

# ===== 輔助功能 =====

def create_cervical_template_lines():
    """建立頸椎範本線"""
    pairs = [("C2", "C3"), ("C3", "C4"), ("C4", "C5"), ("C5", "C6"), ("C6", "C7")]
    
    created_count = 0
    for a, b in pairs:
        for kind in ("lower", "upper"):
            name = f"{a}_{b}_{kind}Endplate"
            
            if slicer.util.getNode(name) is not None:
                continue
            
            ln = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
            ln.AddControlPoint(0, 0, 0)
            ln.AddControlPoint(5, 0, 0)
            ln.SetAttribute("confidence", "0.95")
            ln.SetAttribute("notes", "")
            
            created_count += 1
    
    if created_count > 0:
        slicer.util.infoDisplay(f"Created {created_count} cervical template lines.")
    else:
        slicer.util.infoDisplay("All cervical template lines already exist.")

def validate_annotations():
    """驗證標註完整性"""
    groups = _gatherLines()
    
    if not groups:
        slicer.util.warningDisplay("No annotations found.")
        return
    
    issues = []
    for lvl, parts in groups.items():
        if "lowerEndplate" not in parts:
            issues.append(f"{lvl}: Missing lowerEndplate")
        if "upperEndplate" not in parts:
            issues.append(f"{lvl}: Missing upperEndplate")
    
    if issues:
        slicer.util.warningDisplay("Validation issues found:\n" + "\n".join(issues))
    else:
        slicer.util.infoDisplay(f"Validation passed for {len(groups)} levels.")

# ===== 主執行 =====

if __name__ == "__main__":
    # 在3D Slicer中執行時，這些函數會自動註冊
    print("Spine Annotation Export Script loaded.")
    print("Available functions:")
    print("- export_spine_annotations()")
    print("- create_spine_template_lines()")
    print("- create_cervical_template_lines()")
    print("- validate_annotations()")
