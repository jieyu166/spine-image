# spine_annotation_toolkit.py
# ============================================================
# 3D Slicer Spine Annotation Toolkit
# Complete toolkit for spine endplate annotation workflow
# ============================================================

import json, math, re, datetime, os, sys
import slicer, qt, vtk

# =========================
# Helper Functions
# =========================

_LEVEL_RE = re.compile(r"([CTLS]\d+)[_/]([CTLS]\d+)", re.I)
_C_SUFFIX_RE = re.compile(r"c([0-9]*\.?[0-9]+)", re.I)

def _activeVolumeNode():
    """Get the active background volume from Red slice view"""
    lm = slicer.app.layoutManager()
    if not lm:
        return None
    red = lm.sliceWidget('Red').mrmlSliceCompositeNode()
    vid = red.GetBackgroundVolumeID()
    if not vid:
        return None
    return slicer.util.getNode(vid)

def _getImageInfo(volumeNode):
    """Extract image dimensions and path"""
    img = volumeNode.GetImageData()
    dims = img.GetDimensions() if img else (0,0,0)
    storage = volumeNode.GetStorageNode()
    image_path = storage.GetFileName() if storage and storage.GetFileName() else ""
    return {
        "image_path": image_path,
        "image_dimensions": {"width": int(dims[0]), "height": int(dims[1])}
    }

def _dicomMetaFromNode(volumeNode):
    """Extract DICOM metadata from volume node"""
    meta = {"patient_id":"","study_id":"","study_date":""}
    pid = volumeNode.GetAttribute('DICOM.PatientID')
    suid = volumeNode.GetAttribute('DICOM.StudyInstanceUID')
    sdate = volumeNode.GetAttribute('DICOM.StudyDate')
    if pid: meta["patient_id"]=pid
    if suid: meta["study_id"]=suid
    if sdate and len(sdate)==8:
        meta["study_date"]=f"{sdate[0:4]}-{sdate[4:6]}-{sdate[6:8]}"
    return meta

def _rasToIjk(volumeNode, rasPoint):
    """Convert RAS (world) coordinates to IJK (voxel) coordinates"""
    rasToIjk = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIjk)
    p = list(rasPoint)+[1.0]
    ijk=[0.0,0.0,0.0,0.0]
    rasToIjk.MultiplyPoint(p,ijk)
    return (ijk[0],ijk[1],ijk[2])

def _ijkToRas(volumeNode, i, j, k=0.0):
    """Convert IJK (voxel) to RAS (world) coordinates"""
    ijkToRas = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(ijkToRas)
    p = [float(i), float(j), float(k), 1.0]
    ras = [0.0,0.0,0.0,0.0]
    ijkToRas.MultiplyPoint(p, ras)
    return ras[:3]

def _rasToVec3(p):
    """Convert list/tuple to vtkVector3d"""
    return vtk.vtkVector3d(float(p[0]), float(p[1]), float(p[2]))

def _normalizeLevel(name: str):
    """Extract and normalize level name (e.g., 'L3/L4')"""
    m=_LEVEL_RE.search((name or "").replace(" ",""))
    if not m: return None
    return f"{m.group(1).upper()}/{m.group(2).upper()}"

def _lineTypeFromName(name: str):
    """Determine if line is upper or lower endplate from name"""
    n=(name or "").lower()
    if "lower" in n: return "lowerEndplate"
    if "upper" in n: return "upperEndplate"
    return None

def _angleDegrees(p1,p2):
    """Calculate angle in degrees between two points"""
    dx,dy = p2[0]-p1[0], p2[1]-p1[1]
    return math.degrees(math.atan2(dy,dx))

def _acute(diff_deg: float)->float:
    """Force angle to acute (<=90 degrees)"""
    d = abs(diff_deg) % 180.0
    return 180.0 - d if d > 90.0 else d

def _confidence_from_node(node) -> float:
    """
    Extract confidence value from node
    Priority:
      1) node attribute "confidence" if present
      2) suffix 'C<number>' in node name
      3) default 0.95
    """
    attr = node.GetAttribute("confidence")
    if attr:
        try:
            v = float(attr)
            if 0.0 <= v <= 1.0:
                return v
        except:
            pass
    m = _C_SUFFIX_RE.search(node.GetName() or "")
    if m:
        try:
            v = float(m.group(1))
            if v > 1.0: v = v/100.0 if v <= 100.0 else 1.0
            v = max(0.0, min(1.0, v))
            return v
        except:
            pass
    return 0.95

def _getOrCreateLine(name):
    """Get existing line node or create new one"""
    try:
        return slicer.util.getNode(name)
    except Exception:
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)

def _getOrCreateFiducial(name):
    """Get existing fiducial node or create new one"""
    try:
        return slicer.util.getNode(name)
    except Exception:
        return slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)

def _setLine2ptsWorld(lineNode, p0_ras, p1_ras):
    """Set line node with two world coordinate points"""
    lineNode.RemoveAllControlPoints()
    lineNode.AddControlPointWorld(vtk.vtkVector3d(*p0_ras))
    lineNode.AddControlPointWorld(vtk.vtkVector3d(*p1_ras))

# =========================
# Line Styling
# =========================

def style_line_node(lineNode, point_percent=0.5, line_thickness=1.5, glyph="sphere"):
    """
    Style a markup line node uniformly:
    - point_percent: endpoint size as viewport percentage
    - line_thickness: line width (smaller = thinner)
    - glyph: 'sphere' | 'cross' | 'diamond'
    """
    disp = lineNode.GetDisplayNode()
    if not disp:
        return

    # Line thickness
    try:
        disp.SetLineThickness(float(line_thickness))
    except Exception:
        pass

    # Glyph type
    try:
        t = glyph.lower()
        if t == "sphere":
            disp.SetGlyphType(slicer.vtkMRMLMarkupsDisplayNode.Sphere3D)
        elif t == "cross":
            disp.SetGlyphType(slicer.vtkMRMLMarkupsDisplayNode.StarBurst2D)
        elif t == "diamond":
            disp.SetGlyphType(slicer.vtkMRMLMarkupsDisplayNode.Diamond2D)
    except Exception:
        pass

    # Glyph size (try different APIs for compatibility)
    try:
        disp.SetUseGlyphSizeMm(True)
        disp.SetGlyphSizeMm(2.5)
        return
    except Exception:
        pass

    try:
        disp.SetUseGlyphSizeMm(False)
        disp.SetGlyphScale(float(point_percent) / 100.0 * 10.0)
        return
    except Exception:
        pass

    try:
        disp.SetGlyphSize(float(point_percent))
    except Exception:
        pass

# =========================
# Data Collection
# =========================

def _gatherLines():
    """Collect all markup line nodes and organize by level"""
    groups={}
    for node in slicer.util.getNodesByClass('vtkMRMLMarkupsLineNode'):
        lvl=_normalizeLevel(node.GetName())
        if not lvl: continue
        t=_lineTypeFromName(node.GetName())
        if not t: continue
        if node.GetNumberOfControlPoints()<2: continue
        vol=_activeVolumeNode()
        if not vol: continue

        ras0,ras1=[0,0,0],[0,0,0]
        node.GetNthControlPointPositionWorld(0,ras0)
        node.GetNthControlPointPositionWorld(1,ras1)
        ijk0,ijk1=_rasToIjk(vol,ras0),_rasToIjk(vol,ras1)
        p0={"x":float(ijk0[0]),"y":float(ijk0[1])}
        p1={"x":float(ijk1[0]),"y":float(ijk1[1])}
        
        confidence = _confidence_from_node(node)
        notes = node.GetAttribute("notes") or ""

        if lvl not in groups: groups[lvl]={}
        groups[lvl][t]=(p0,p1,confidence,notes)
    return groups

# =========================
# Build Vertebra Edges
# =========================

def _build_vertebra_edges(measurements):
    """
    Build vertebra boundaries from adjacent endplates:
    - For level A/B (e.g., L4/L5), generate vertebra edge for 'A' (L4)
    - Use lowerEndplate from previous level X/A (e.g., L3/L4)
    - Match with upperEndplate from current level A/B (e.g., L4/L5)
    - For each endplate: min x = anterior edge; max x = posterior edge
    - Output: vertebra_edges[A] = { anterior:[..,.], posterior:[..,.] }
    """
    level_dict = {m["level"]: m for m in measurements}
    
    # Reverse index to find 'previous level' (where second == A)
    prev_by_inf = {}
    for k in level_dict.keys():
        sup, inf = k.split("/")
        prev_by_inf[inf.upper()] = k  # e.g. prev_by_inf["L4"] = "L3/L4"

    vertebra_edges = {}

    for cur_level, cur_m in level_dict.items():
        sup, inf = cur_level.split("/")
        sup = sup.upper()  # A
        inf = inf.upper()  # B

        # Find previous level: X/A
        prev_level = prev_by_inf.get(sup)
        if not prev_level:
            continue

        prev_m = level_dict.get(prev_level)
        if not prev_m:
            continue

        # Need: previous level's lowerEndplate, current level's upperEndplate
        if "lowerEndplate" not in prev_m or "upperEndplate" not in cur_m:
            continue

        prev_lower_line = prev_m["lowerEndplate"]
        cur_upper_line  = cur_m["upperEndplate"]

        # min x = anterior edge; max x = posterior edge
        prev_lower_ant = min(prev_lower_line, key=lambda p: p["x"])
        prev_lower_post = max(prev_lower_line, key=lambda p: p["x"])

        cur_upper_ant = min(cur_upper_line, key=lambda p: p["x"])
        cur_upper_post = max(cur_upper_line, key=lambda p: p["x"])

        vertebra_edges[sup] = {
            "anterior": [prev_lower_ant, cur_upper_ant],
            "posterior": [prev_lower_post, cur_upper_post]
        }

    return vertebra_edges

# =========================
# FUNCTION 1: Export Annotations
# =========================

def export_spine_annotations():
    """
    Export spine endplate annotations to JSON format
    - Collects all markup lines
    - Shows UI dialog for metadata
    - Includes surgery information
    - Calculates angles and confidence
    - Builds vertebra edges
    """
    vol=_activeVolumeNode()
    if not vol:
        slicer.util.errorDisplay("No background volume found.")
        return
    groups=_gatherLines()
    if not groups:
        slicer.util.errorDisplay("No MarkupsLine nodes found.")
        return

    dicomMeta=_dicomMetaFromNode(vol)
    imageInfo=_getImageInfo(vol)

    # ---------- UI Dialog ----------
    d=qt.QDialog(slicer.util.mainWindow())
    d.setWindowTitle("Spine Metadata")
    mainLayout=qt.QVBoxLayout(d)
    formLayout=qt.QFormLayout()

    spineType=qt.QComboBox()
    spineType.addItems(["L","C","T","S"])
    imageType=qt.QComboBox()
    imageType.addItems(["flexion","extension","neutral"])
    annotName=qt.QLineEdit()
    annotId=qt.QLineEdit()
    annotSpec=qt.QLineEdit()
    annotYrs=qt.QLineEdit()

    # Surgery controls
    surgeryCheck = qt.QCheckBox("Surgery done")
    surgeryType  = qt.QLineEdit()
    surgeryType.setPlaceholderText("e.g., posterior instrumentation with interbody cage")
    surgeryLvls  = qt.QLineEdit()
    surgeryLvls.setPlaceholderText("e.g., L2, L3, L4, L5")

    report=qt.QPlainTextEdit()
    report.setPlaceholderText("Paste original clinical report...")
    report.setFixedHeight(60)
    report.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

    formLayout.addRow("Spine type",spineType)
    formLayout.addRow("Image type",imageType)
    formLayout.addRow("Annotator name",annotName)
    formLayout.addRow("Annotator ID",annotId)
    formLayout.addRow("Specialty",annotSpec)
    formLayout.addRow("Experience years",annotYrs)
    formLayout.addRow(surgeryCheck)
    formLayout.addRow("Surgery type (optional)", surgeryType)
    formLayout.addRow("Surgery levels (optional)", surgeryLvls)
    formLayout.addRow("Original report",report)
    mainLayout.addLayout(formLayout)

    # Buttons
    okButton = qt.QPushButton("OK")
    cancelButton = qt.QPushButton("Cancel")
    okButton.clicked.connect(lambda checked=False: d.accept())
    cancelButton.clicked.connect(lambda checked=False: d.reject())
    btnLayout = qt.QHBoxLayout()
    btnLayout.addStretch(1)
    btnLayout.addWidget(okButton)
    btnLayout.addWidget(cancelButton)
    mainLayout.addLayout(btnLayout)

    d.setLayout(mainLayout)
    d.setMinimumWidth(480)
    d.setMinimumHeight(420)

    if d.exec_() != qt.QDialog.Accepted:
        return

    # ---------- Build JSON ----------
    out={
        "patient_id":dicomMeta["patient_id"],
        "study_id":dicomMeta["study_id"],
        "study_date":dicomMeta["study_date"],
        "spine_type":spineType.currentText,
        "image_type":imageType.currentText,
        "image_path":imageInfo["image_path"],
        "image_dimensions":imageInfo["image_dimensions"],
        "annotator":{
            "name":annotName.text.strip(),
            "id":annotId.text.strip(),
            "specialty":annotSpec.text.strip(),
            "experience_years": int(annotYrs.text) if annotYrs.text.strip().isdigit() else None
        },
        "annotation_date":datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measurements":[],
        "clinical_notes":{"original_report":report.toPlainText()}
    }

    # Surgery info
    surgery_info = {"surgery_done": bool(surgeryCheck.isChecked())}
    stype = surgeryType.text.strip()
    if stype:
        surgery_info["surgery_type"] = stype
    slvls = surgeryLvls.text.strip()
    if slvls:
        raw = re.split(r"[,\s/]+", slvls)
        lvls = [t for t in (s.strip().upper() for s in raw) if t]
        surgery_info["surgery_levels"] = lvls
    out["surgery_info"] = surgery_info

    # ---------- Measurements ----------
    for lvl, parts in groups.items():
        if "lowerEndplate" not in parts or "upperEndplate" not in parts:
            continue
        l0,l1,confL,notesL=parts["lowerEndplate"]
        u0,u1,confU,notesU=parts["upperEndplate"]
        a1=_angleDegrees((l0["x"],l0["y"]),(l1["x"],l1["y"]))
        a2=_angleDegrees((u0["x"],u0["y"]),(u1["x"],u1["y"]))
        raw=abs(a1-a2)%180.0
        ang=_acute(raw)

        entry={
            "level":lvl,
            "angle":round(ang,1),
            "angle_raw":round(raw,1),
            "confidence":round((confL+confU)/2,3),
            "measurement_method":"manual",
            "lowerEndplate":[l0,l1],
            "upperEndplate":[u0,u1]
        }
        notes = "; ".join([x for x in (notesL,notesU) if x])
        if notes:
            entry["measurement_notes"]=notes
        out["measurements"].append(entry)

    # ---------- Vertebra edges ----------
    out["vertebra_edges"] = _build_vertebra_edges(out["measurements"])

    # ---------- Save ----------
    path = qt.QFileDialog.getSaveFileName(
        slicer.util.mainWindow(),
        "Save Annotations JSON",
        "",
        "JSON Files (*.json)"
    )
    if not path: return
    with open(path,"w",encoding="utf-8") as f:
        json.dump(out,f,ensure_ascii=False,indent=2)
    slicer.util.infoDisplay(f"Exported {len(out['measurements'])} levels to:\n{path}")

# =========================
# FUNCTION 2: Create Template Lines
# =========================

def create_spine_template_lines(length_ratio=0.25):
    """
    Generate template lines for L1-S1 at image center
    Each line length = image width * length_ratio (default 25%)
    Lines positioned on visible slice plane
    """
    vol = _activeVolumeNode()
    if not vol:
        slicer.util.errorDisplay("No background volume found.")
        return

    img = vol.GetImageData()
    dims = img.GetDimensions() if img else (0,0,0)
    cols, rows, slices = dims[0], dims[1], max(1, dims[2])
    k = (slices - 1) * 0.5  # middle slice

    # Center point (IJK)
    cx, cy = cols * 0.5, rows * 0.5
    half_len = max(5.0, cols * float(length_ratio) * 0.5)
    v_step = rows * 0.06
    start_y = rows * 0.30

    pairs = [("L1","L2"),("L2","L3"),("L3","L4"),("L4","L5"),("L5","S1")]
    kinds = ("lower","upper")
    y_offset = {"lower": -v_step*0.25, "upper": +v_step*0.25}

    created = 0
    for idx, (a,b) in enumerate(pairs):
        base_y = start_y + idx * v_step
        for kind in kinds:
            name = f"{a}_{b}_{kind}"
            # Skip if already exists
            try:
                slicer.util.getNode(name)
                continue
            except Exception:
                pass

            # IJK endpoints (horizontal line)
            y = base_y + y_offset[kind]
            p0_ijk = (cx - half_len, y, k)
            p1_ijk = (cx + half_len, y, k)

            p0_ras = _ijkToRas(vol, *p0_ijk)
            p1_ras = _ijkToRas(vol, *p1_ijk)

            ln = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
            ln.RemoveAllControlPoints()
            ln.AddControlPointWorld(_rasToVec3(p0_ras))
            ln.AddControlPointWorld(_rasToVec3(p1_ras))

            # Style: large points, thin line
            style_line_node(ln, point_percent=0.5, line_thickness=1.5, glyph="sphere")

            # Default attributes
            ln.SetAttribute("confidence","0.95")
            ln.SetAttribute("notes","")

            # Ensure editable
            ln.SetLocked(False)
            ln.SetDisplayVisibility(True)
            created += 1

    slicer.util.infoDisplay(f"Created {created} template lines at image center (length_ratio={length_ratio}).")

# =========================
# FUNCTION 3: Import Annotations
# =========================

def import_spine_annotations(json_path=None, draw_vertebra_edges=True, k_slice=0.0):
    """
    Import spine annotations JSON into Slicer:
    - Creates lines for each measurement: <Lx_Ly>_lower / _upper
    - Restores landmarks as Fiducials: LM_<Lx_Ly>
    - (Optional) Draws vertebra_edges as lines: VE_<Lv>_anterior / posterior
    
    Parameters:
      json_path: Path to JSON file (None prompts file dialog)
      draw_vertebra_edges: Whether to draw vertebra edge lines
      k_slice: IJK k value; typically 0 for 2D images
    """
    vol = _activeVolumeNode()
    if not vol:
        slicer.util.errorDisplay("Please load an image in Red view as Background Volume first.")
        return

    # File selection
    if not json_path:
        json_path = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(), 
            "Open Annotations JSON", 
            "", 
            "JSON Files (*.json)")
        if not json_path:
            return

    # Read JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ms = data.get("measurements", [])
    
    # Restore lower/upper lines
    for m in ms:
        level = m["level"]  # e.g., "L3/L4"
        lo = m.get("lowerEndplate")
        up = m.get("upperEndplate")
        
        # Create lower line
        if lo and len(lo) >= 2:
            name = level.replace("/", "_") + "_lower"
            ln = _getOrCreateLine(name)
            conf = m.get("confidence", 0.95)
            ln.SetAttribute("confidence", str(conf))
            if "measurement_notes" in m:
                ln.SetAttribute("notes", m["measurement_notes"])
            p0 = _ijkToRas(vol, lo[0]["x"], lo[0]["y"], k_slice)
            p1 = _ijkToRas(vol, lo[1]["x"], lo[1]["y"], k_slice)
            _setLine2ptsWorld(ln, p0, p1)

        # Create upper line
        if up and len(up) >= 2:
            name = level.replace("/", "_") + "_upper"
            ln = _getOrCreateLine(name)
            conf = m.get("confidence", 0.95)
            ln.SetAttribute("confidence", str(conf))
            if "measurement_notes" in m:
                ln.SetAttribute("notes", m["measurement_notes"])
            p0 = _ijkToRas(vol, up[0]["x"], up[0]["y"], k_slice)
            p1 = _ijkToRas(vol, up[1]["x"], up[1]["y"], k_slice)
            _setLine2ptsWorld(ln, p0, p1)

        # Restore landmarks (optional)
        if "landmarks" in m and m["landmarks"]:
            fid = _getOrCreateFiducial("LM_" + level.replace("/", "_"))
            for lm in m["landmarks"]:
                pr = _ijkToRas(vol, lm["x"], lm["y"], k_slice)
                idx = fid.AddControlPointWorld(vtk.vtkVector3d(*pr))
                label = lm.get("type", "landmark")
                fid.SetNthControlPointLabel(idx, label)

    # Restore vertebra edges (optional)
    if draw_vertebra_edges and "vertebra_edges" in data:
        for lv, ed in data["vertebra_edges"].items():
            # Anterior edge
            ant = ed.get("anterior")
            if ant and len(ant) == 2:
                name = f"VE_{lv}_anterior"
                ln = _getOrCreateLine(name)
                p0 = _ijkToRas(vol, ant[0]["x"], ant[0]["y"], k_slice)
                p1 = _ijkToRas(vol, ant[1]["x"], ant[1]["y"], k_slice)
                _setLine2ptsWorld(ln, p0, p1)
            
            # Posterior edge
            post = ed.get("posterior")
            if post and len(post) == 2:
                name = f"VE_{lv}_posterior"
                ln = _getOrCreateLine(name)
                p0 = _ijkToRas(vol, post[0]["x"], post[0]["y"], k_slice)
                p1 = _ijkToRas(vol, post[1]["x"], post[1]["y"], k_slice)
                _setLine2ptsWorld(ln, p0, p1)

    slicer.util.infoDisplay(f"Imported annotations from:\n{json_path}")

# =========================
# FUNCTION 4: View JSON Overlay
# =========================

def _is_slicer():
    """Check if running in Slicer environment"""
    try:
        import slicer  # noqa
        return True
    except Exception:
        return False

def load_json(path):
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sanity_checks(dicom_size, json_data):
    """Compare DICOM size with JSON dimensions; return warning list"""
    warns = []
    dim = json_data.get("image_dimensions", {})
    jcols, jrows = dim.get("width"), dim.get("height")
    if dicom_size and jcols and jrows:
        cols, rows = dicom_size
        if (cols != jcols) or (rows != jrows):
            warns.append(f"Size mismatch: DICOM {cols}x{rows} vs JSON {jcols}x{jrows}")
    levels = [m.get("level") for m in json_data.get("measurements", [])]
    if not levels:
        warns.append("JSON contains no measurements.")
    if len(levels) != len(set(levels)):
        warns.append("JSON contains duplicate levels.")
    return warns

def _slicer_view(dicom_path=None, json_path=None):
    """Slicer version of view_spine_json_overlay"""
    def pick_file(caption, flt):
        return qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), caption, "", flt)

    # Get files
    if not dicom_path:
        dicom_path = pick_file("Open DICOM", "DICOM (*.dcm);;All (*)")
        if not dicom_path:
            return
    if not json_path:
        json_path = pick_file("Open JSON", "JSON (*.json);;All (*)")
        if not json_path:
            return

    # Load DICOM as Volume
    vol = slicer.util.loadVolume(dicom_path)
    if not vol:
        slicer.util.errorDisplay("DICOM loading failed")
        return

    # Display in Red view and fit
    lm = slicer.app.layoutManager()
    sw = lm.sliceWidget("Red")
    comp = sw.mrmlSliceCompositeNode()
    comp.SetBackgroundVolumeID(vol.GetID())
    sw.fitSliceToBackground()

    # Read JSON
    data = load_json(json_path)

    # Size check
    img = vol.GetImageData()
    dims = img.GetDimensions() if img else (0,0,0)
    warns = sanity_checks((dims[0], dims[1]), data)

    def ijk_to_ras(i, j, k=0.0):
        mat = vtk.vtkMatrix4x4()
        vol.GetIJKToRASMatrix(mat)
        p = [float(i), float(j), float(k), 1.0]
        out = [0.0,0.0,0.0,0.0]
        mat.MultiplyPoint(p, out)
        return out[:3]

    def new_line(name, rgb=(0.2,0.45,0.95), thick=1.5):
        ln = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", name)
        disp = ln.GetDisplayNode()
        disp.SetSelectedColor(*rgb)
        disp.SetColor(*rgb)
        disp.SetLineThickness(thick)
        # Set smaller glyph size for precise selection
        try:
            disp.SetUseGlyphSizeMm(True)
            disp.SetGlyphSizeMm(1.5)  # Small precise points
        except:
            try:
                disp.SetUseGlyphSizeMm(False)
                disp.SetGlyphScale(0.3)  # Small scale
            except:
                pass
        return ln

    def set_line_world(ln, p0, p1):
        ln.RemoveAllControlPoints()
        ln.AddControlPointWorld(vtk.vtkVector3d(*p0))
        ln.AddControlPointWorld(vtk.vtkVector3d(*p1))

    # Overlay endplates with thinner lines
    for m in data.get("measurements", []):
        level = m.get("level","?")
        lo = m.get("lowerEndplate")
        up = m.get("upperEndplate")

        if lo and len(lo)==2:
            name = f"{level.replace('/','_')}_lower_VIEW"
            ln = new_line(name, rgb=(0.2,0.45,0.95), thick=1.5)
            p0 = ijk_to_ras(lo[0]["x"], lo[0]["y"], 0.0)
            p1 = ijk_to_ras(lo[1]["x"], lo[1]["y"], 0.0)
            set_line_world(ln, p0, p1)

        if up and len(up)==2:
            name = f"{level.replace('/','_')}_upper_VIEW"
            ln = new_line(name, rgb=(0.98,0.55,0.2), thick=1.5)
            p0 = ijk_to_ras(up[0]["x"], up[0]["y"], 0.0)
            p1 = ijk_to_ras(up[1]["x"], up[1]["y"], 0.0)
            set_line_world(ln, p0, p1)

    # Draw vertebra_edges (if present)
    if "vertebra_edges" in data:
        thin_flex = (0.2,0.45,0.95)
        thin_ext  = (0.98,0.55,0.2)
        
        def draw_edge(name, pts, color):
            if not (pts and len(pts)==2): return
            ln = new_line(name, rgb=color, thick=2.0)
            p0 = ijk_to_ras(pts[0]["x"], pts[0]["y"], 0.0)
            p1 = ijk_to_ras(pts[1]["x"], pts[1]["y"], 0.0)
            set_line_world(ln, p0, p1)

        for vtx, ed in data["vertebra_edges"].items():
            ant = ed.get("anterior")
            post = ed.get("posterior")
            draw_edge(f"VE_{vtx}_ant_VIEW", ant, thin_flex)
            draw_edge(f"VE_{vtx}_post_VIEW", post, thin_ext)

    # Display warnings
    if warns:
        slicer.util.infoDisplay("Check warnings:\n- " + "\n- ".join(warns), 
                                windowTitle="JSON / DICOM Check")
    else:
        slicer.util.infoDisplay("Size and content check passed.", 
                                windowTitle="JSON / DICOM Check")

def _py_view(dicom_path, json_path):
    """Pure Python version for Colab/standalone"""
    try:
        import pydicom, numpy as np, matplotlib.pyplot as plt
    except Exception:
        print("Installing dependencies: pydicom matplotlib numpy ...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                              "pydicom", "matplotlib", "numpy"])
        import pydicom, numpy as np, matplotlib.pyplot as plt

    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(float)
    if img.ptp() > 0:
        img = (img - img.min()) / (img.max() - img.min())

    data = load_json(json_path)

    # Size check
    rows = int(getattr(ds, "Rows", img.shape[0]))
    cols = int(getattr(ds, "Columns", img.shape[1]))
    warns = sanity_checks((cols, rows), data)
    if warns:
        print("⚠️ Check warnings:")
        for w in warns:
            print(" -", w)

    def draw_line(ax, p0, p1, color, lw=2.5, alpha=0.95, label=None):
        ax.plot([p0["x"], p1["x"]], [p0["y"], p1["y"]], 
               '-', color=color, linewidth=lw, alpha=alpha)
        if label:
            ax.text(p0["x"], p0["y"], label, color=color, fontsize=8)

    fig, ax = plt.subplots(1, 1, figsize=(7, 10))
    ax.imshow(img, cmap='gray', origin='upper')
    ax.set_title("DICOM + JSON Overlay")
    ax.set_axis_off()

    # Draw lower (blue) / upper (orange)
    for m in data.get("measurements", []):
        lvl = m.get("level","?")
        lo = m.get("lowerEndplate")
        up = m.get("upperEndplate")
        if lo and len(lo)==2:
            draw_line(ax, lo[0], lo[1], color='tab:blue', lw=2.5, 
                     label=f"{lvl}-lower")
        if up and len(up)==2:
            draw_line(ax, up[0], up[1], color='tab:orange', lw=2.5, 
                     label=f"{lvl}-upper")

    # Draw vertebra_edges (if present)
    if "vertebra_edges" in data:
        for vtx, ed in data["vertebra_edges"].items():
            ant = ed.get("anterior")
            post = ed.get("posterior")
            if ant and len(ant)==2:
                draw_line(ax, ant[0], ant[1], color='tab:blue', lw=1.8, 
                         alpha=0.7, label=f"{vtx}-ant")
            if post and len(post)==2:
                draw_line(ax, post[0], post[1], color='tab:orange', lw=1.8, 
                         alpha=0.7, label=f"{vtx}-post")

    plt.tight_layout()
    plt.show()

def view_spine_json_overlay(dicom_path=None, json_path=None):
    """
    View spine JSON overlay on DICOM image
    Works in both Slicer and standalone Python/Colab environments
    """
    if _is_slicer():
        _slicer_view(dicom_path, json_path)
    else:
        if not dicom_path or not json_path:
            print("Usage: python spine_annotation_toolkit.py --dicom /path/to/image.dcm --json /path/to/annotation.json")
            return
        _py_view(dicom_path, json_path)

# =========================
# Command Line Interface
# =========================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Spine Annotation Toolkit")
    ap.add_argument("--dicom", dest="dicom_path", help="Path to DICOM file")
    ap.add_argument("--json", dest="json_path", help="Path to JSON file")
    args = ap.parse_args()
    view_spine_json_overlay(dicom_path=args.dicom_path, json_path=args.json_path)

# =========================
# Usage Instructions
# =========================
"""
SPINE ANNOTATION TOOLKIT - 3D Slicer
====================================

In 3D Slicer Python Console, run:

1. CREATE TEMPLATE LINES (for annotation)
   >>> create_spine_template_lines(length_ratio=0.25)

2. EXPORT ANNOTATIONS TO JSON
   >>> export_spine_annotations()

3. IMPORT JSON BACK TO SLICER
   >>> import_spine_annotations()

4. VIEW JSON OVERLAY (for verification)
   >>> view_spine_json_overlay()

Command line (outside Slicer):
   python spine_annotation_toolkit.py --dicom image.dcm --json annotation.json
"""