from __future__ import annotations

import cgi
import html
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class AnalysisResult:
    image_width: int | None
    image_height: int | None
    vertebral_bodies: list[str]
    endplates: list[str]
    discs: list[str]
    listhesis: list[str]
    warnings: list[str]


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def analyze_spine_image(_: Path) -> AnalysisResult:
    warnings = [
        "目前僅建立網頁介面與分析流程骨架，尚未接入AI模型/影像辨識。",
        "請在後端整合椎體分割、終板偵測與量測模型後再提供臨床判讀結果。",
    ]
    return AnalysisResult(
        image_width=None,
        image_height=None,
        vertebral_bodies=[],
        endplates=[],
        discs=[],
        listhesis=[],
        warnings=warnings,
    )


def render_index(error: str | None = None) -> str:
    error_html = f"<div class=\"alert\">{html.escape(error)}</div>" if error else ""
    return f"""
<!doctype html>
<html lang=\"zh-Hant\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Spine X-ray 分析</title>
    <link rel=\"stylesheet\" href=\"/static/styles.css\" />
  </head>
  <body>
    <main class=\"container\">
      <header>
        <h1>Spine X-ray 分析</h1>
        <p class=\"subtitle\">上傳 spine X-ray，自動抓取 vertebral border 並產出 disc/endplate 量測結果。</p>
      </header>
      {error_html}
      <section class=\"card\">
        <h2>上傳影像</h2>
        <form class=\"upload-form\" action=\"/analyze\" method=\"post\" enctype=\"multipart/form-data\">
          <label class=\"file-input\">
            <input type=\"file\" name=\"image\" accept=\"image/*\" required />
            <span>選擇 X-ray 影像</span>
          </label>
          <button type=\"submit\">開始分析</button>
        </form>
        <ul class=\"notes\">
          <li>支援 JPG / PNG / TIFF / BMP。</li>
          <li>後端尚未接入 AI 模型，會顯示流程骨架與提醒。</li>
        </ul>
      </section>
      <section class=\"card\">
        <h2>分析流程</h2>
        <ol class=\"steps\">
          <li>辨識每個 vertebral body。</li>
          <li>辨識每個 vertebral body endplates。</li>
          <li>計算 disc 上/下 endplate 夾角。</li>
          <li>量測 disc 平均高度並檢查是否 narrowing。</li>
          <li>檢查後緣 endplate 對齊以判斷 spondylolisthesis。</li>
        </ol>
      </section>
    </main>
  </body>
</html>
"""


def render_result(filename: str, result: AnalysisResult) -> str:
    width = result.image_width if result.image_width is not None else "未知"
    height = result.image_height if result.image_height is not None else "未知"
    warning_items = "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings)
    return f"""
<!doctype html>
<html lang=\"zh-Hant\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>分析結果</title>
    <link rel=\"stylesheet\" href=\"/static/styles.css\" />
  </head>
  <body>
    <main class=\"container\">
      <header class=\"header-row\">
        <div>
          <h1>分析結果</h1>
          <p class=\"subtitle\">影像尺寸：{width} x {height} px</p>
        </div>
        <a class=\"link\" href=\"/\">重新上傳</a>
      </header>
      <section class=\"card\">
        <h2>影像預覽</h2>
        <div class=\"preview\">
          <img src=\"/uploads/{html.escape(filename)}\" alt=\"X-ray preview\" />
        </div>
      </section>
      <section class=\"grid\">
        <div class=\"card\">
          <h2>Vertebral bodies</h2>
          <p class=\"empty\">尚未偵測到椎體資料。</p>
        </div>
        <div class=\"card\">
          <h2>Endplates</h2>
          <p class=\"empty\">尚未偵測到 endplate。</p>
        </div>
        <div class=\"card\">
          <h2>Disc 量測</h2>
          <p class=\"empty\">尚未計算 disc 角度或高度。</p>
        </div>
        <div class=\"card\">
          <h2>Listhesis 檢查</h2>
          <p class=\"empty\">尚未偵測到位移資料。</p>
        </div>
      </section>
      <section class=\"card\">
        <h2>提醒</h2>
        <ul>
          {warning_items}
        </ul>
      </section>
    </main>
  </body>
</html>
"""


class SpineRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_index().encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/analyze":
            self.send_error(404, "Not Found")
            return

        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        if "image" not in form:
            self._respond_html(render_index("未找到上傳檔案。"))
            return

        field = form["image"]
        if not field.filename:
            self._respond_html(render_index("請選擇影像檔案。"))
            return

        if not allowed_file(field.filename):
            self._respond_html(render_index("檔案格式不支援，請上傳 JPG/PNG/TIFF/BMP 影像。"))
            return

        UPLOAD_DIR.mkdir(exist_ok=True)
        extension = Path(field.filename).suffix.lower()
        filename = f"{uuid.uuid4().hex}{extension}"
        filepath = UPLOAD_DIR / filename

        with filepath.open("wb") as target:
            data = field.file.read()
            target.write(data)

        result = analyze_spine_image(filepath)
        self._respond_html(render_result(filename, result))

    def _respond_html(self, body: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))


def run_server() -> None:
    port = int(os.environ.get("PORT", "5000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), SpineRequestHandler)
    print(f"Server running on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
