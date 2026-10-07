import sys
import os
import time
import json
import base64
import requests
import re
import numpy as np
from PySide6.QtCore import Qt, QRect, QPoint, Signal, Slot, QObject, QThread, QByteArray, QBuffer, QIODevice, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QRegion, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QComboBox, QTextEdit, QScrollArea,
                               QProgressBar, QFrame, QGridLayout)
from PySide6.QtMultimedia import QScreenCapture, QMediaCaptureSession, QVideoSink, QVideoFrame

OLLAMA_URL = "http://127.0.0.1:11434/api"

MODEL_CATALOG = [
    {"id": "qwen3-vl", "name": "Qwen3-VL", "speed": "Fast", "accuracy": "Highest (Best OCR)", "desc": "Flagship Qwen vision model. Best for text/screenshots."},
    {"id": "pixtral", "name": "Pixtral (12B)", "speed": "Medium", "accuracy": "Very High", "desc": "Mistral's heavy vision model. Excellent reasoning and instruction following."},
    {"id": "gemma4", "name": "Gemma 4", "speed": "Fast", "accuracy": "High", "desc": "Google's multimodal vision model."},
    {"id": "qwen2.5-vl", "name": "Qwen2.5-VL", "speed": "Fast", "accuracy": "High", "desc": "Strong OCR and visual understanding."},
    {"id": "minicpm-v", "name": "MiniCPM-V", "speed": "Very Fast", "accuracy": "Medium", "desc": "Mobile-sized model, very low RAM footprint."},
    {"id": "llava:7b", "name": "LLaVA 7B", "speed": "Medium", "accuracy": "Medium", "desc": "Standard, balanced vision architecture (safest fallback)."}
]

MODES = {
    "Trivia Solver": (
        "You are an expert trivia solver. Look at the image containing a question and multiple-choice options.\n"
        "Step 1: Transcribe the question exactly.\n"
        "Step 2: Transcribe ALL the options provided in the image.\n"
        "Step 3: Stop and think. Search your internal knowledge base for the specific media, historical event, or fact mentioned. Do not guess based on what sounds right.\n"
        "Step 4: State the final correct option.\n"
        "Format EXACTLY as:\n"
        "Question: <transcribe question>\n"
        "Options: <list all options>\n"
        "Reason: <1-2 sentences proving you know the exact factual background before picking an option>\n"
        "Answer: <The exact correct option text>"
    ),
    "General Opinion": "Look at the content in this image and provide a concise, 1-sentence opinion or direct answer.",
    "Summarize": "Summarize the text visible in this image in one short sentence."
}


class SpectacleOverlay(QWidget):
    box_changed = Signal(QRect)
    first_selection_done = Signal(QRect)
    selection_canceled = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)

        self.mode = 'hidden'
        self.margin = 12
        self.min_size = 40

        self.drag_start = None
        self.drag_current = None
        self.box_rect = QRect()

        self.resize_action = None
        self.resize_start_pos = None
        self.resize_start_rect = None

    def start_selection(self):
        self.mode = 'selecting'
        self.clearMask()
        self.drag_start = None
        self.drag_current = None
        self.box_rect = QRect()
        self.setCursor(Qt.CrossCursor)

        screen_geom = QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geom)
        self.showNormal()
        self.update()

    def update_mask_for_active_box(self):
        m = self.margin
        outer = QRegion(self.box_rect.adjusted(-m, -m, m, m))
        self.setMask(outer)

    def get_action_at(self, pos):
        m = self.margin
        r = self.box_rect
        x, y = pos.x(), pos.y()

        if not r.adjusted(-m, -m, m, m).contains(pos):
            return None

        l = abs(x - r.left()) <= m
        ri = abs(x - r.right()) <= m
        t = abs(y - r.top()) <= m
        b = abs(y - r.bottom()) <= m

        if l and t: return 'tl'
        if ri and t: return 'tr'
        if l and b: return 'bl'
        if ri and b: return 'br'
        if l: return 'l'
        if ri: return 'r'
        if t: return 't'
        if b: return 'b'

        return 'move'

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        pos = event.position().toPoint()
        if self.mode == 'selecting':
            self.drag_start = pos
            self.drag_current = pos
            self.update()
        elif self.mode == 'active':
            self.resize_action = self.get_action_at(pos)
            if self.resize_action:
                self.clearMask()
                self.resize_start_pos = pos
                self.resize_start_rect = QRect(self.box_rect)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()

        if self.mode == 'selecting':
            if self.drag_start:
                self.drag_current = pos
                self.box_rect = QRect(self.drag_start, self.drag_current).normalized()
                self.update()
        elif self.mode == 'active':
            if not self.resize_action:
                action = self.get_action_at(pos)
                if action in ('tl', 'br'): self.setCursor(Qt.SizeFDiagCursor)
                elif action in ('tr', 'bl'): self.setCursor(Qt.SizeBDiagCursor)
                elif action in ('l', 'r'): self.setCursor(Qt.SizeHorCursor)
                elif action in ('t', 'b'): self.setCursor(Qt.SizeVerCursor)
                elif action == 'move': self.setCursor(Qt.SizeAllCursor)
                else: self.setCursor(Qt.ArrowCursor)
                return

            delta = pos - self.resize_start_pos
            g = QRect(self.resize_start_rect)

            if self.resize_action == 'move':
                g.moveTo(g.topLeft() + delta)
            else:
                if 'l' in self.resize_action: g.setLeft(min(g.left() + delta.x(), g.right() - self.min_size))
                if 'r' in self.resize_action: g.setRight(max(g.right() + delta.x(), g.left() + self.min_size))
                if 't' in self.resize_action: g.setTop(min(g.top() + delta.y(), g.bottom() - self.min_size))
                if 'b' in self.resize_action: g.setBottom(max(g.bottom() + delta.y(), g.top() + self.min_size))

            old_rect = QRect(self.box_rect)
            self.box_rect = g.normalized()

            m = self.margin
            bounding_area = old_rect.united(self.box_rect).adjusted(-m, -m, m, m)
            self.setMask(self.mask().united(QRegion(bounding_area)))
            self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == 'selecting':
            if self.drag_start and self.drag_current:
                final_r = QRect(self.drag_start, self.drag_current).normalized()
                if final_r.width() > 30 and final_r.height() > 30:
                    self.mode = 'active'
                    self.box_rect = final_r

                    self.clearMask()
                    self.update()
                    QTimer.singleShot(100, self.update_mask_for_active_box)

                    self.first_selection_done.emit(self.box_rect)
                else:
                    self.selection_canceled.emit()
            else:
                self.selection_canceled.emit()
        elif self.mode == 'active':
            self.resize_action = None
            self.update_mask_for_active_box()
            self.box_changed.emit(self.box_rect)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.selection_canceled.emit()
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

        if self.mode == 'selecting':
            painter.fillRect(self.rect(), QColor(0, 0, 0, 140))

            if not self.drag_start:
                banner_w, banner_h = 420, 64
                bx = (self.width() - banner_w) // 2
                by = self.height() // 5
                banner_rect = QRect(bx, by, banner_w, banner_h)

                painter.setBrush(QColor(30, 32, 36, 230))
                painter.setPen(QPen(QColor(61, 174, 233, 200), 1.5))
                painter.drawRoundedRect(banner_rect, 8, 8)

                painter.setPen(QColor(255, 255, 255))
                font_title = QFont()
                font_title.setPointSize(13)
                font_title.setBold(True)
                painter.setFont(font_title)
                painter.drawText(QRect(bx, by + 8, banner_w, 24), Qt.AlignCenter, "Click and drag to select an area")

                painter.setPen(QColor(170, 170, 170))
                font_sub = QFont()
                font_sub.setPointSize(10)
                painter.setFont(font_sub)
                painter.drawText(QRect(bx, by + 34, banner_w, 20), Qt.AlignCenter, "Press Esc to cancel")

            if self.drag_start and self.drag_current and not self.box_rect.isEmpty():
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.fillRect(self.box_rect, Qt.transparent)
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

                painter.setPen(QPen(QColor("#3daee9"), 2))
                painter.drawRect(self.box_rect)

        elif self.mode == 'active':
            if not self.box_rect.isEmpty():
                painter.setPen(QPen(QColor("#3daee9"), 3))
                painter.drawRect(self.box_rect)

                painter.setBrush(QColor("#ffffff"))
                painter.setPen(QPen(QColor("#3daee9"), 2))
                h_size = 8
                pts = [
                    self.box_rect.topLeft(), self.box_rect.topRight(),
                    self.box_rect.bottomLeft(), self.box_rect.bottomRight(),
                    QPoint(self.box_rect.center().x(), self.box_rect.top()),
                    QPoint(self.box_rect.center().x(), self.box_rect.bottom()),
                    QPoint(self.box_rect.left(), self.box_rect.center().y()),
                    QPoint(self.box_rect.right(), self.box_rect.center().y())
                ]
                for p in pts:
                    painter.drawEllipse(p, h_size // 2, h_size // 2)


class ModelDownloader(QObject):
    progress_update = Signal(str, int)
    finished = Signal(bool)

    @Slot(str)
    def pull(self, model_id):
        try:
            self.progress_update.emit(f"Connecting to pull {model_id}...", 0)
            res = requests.post(f"{OLLAMA_URL}/pull", json={"name": model_id, "stream": True}, stream=True, timeout=10)
            if res.status_code != 200:
                self.progress_update.emit(f"Server error: {res.status_code}", 0)
                self.finished.emit(False)
                return

            for line in res.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    status = data.get("status", "")
                    if "completed" in data and "total" in data:
                        completed = data["completed"]
                        total = data["total"]
                        pct = int((completed / total) * 100)
                        comp_gb = completed / (1024**3)
                        tot_gb = total / (1024**3)
                        self.progress_update.emit(f"{status.capitalize()} - {comp_gb:.1f} GB / {tot_gb:.1f} GB", pct)
                    else:
                        self.progress_update.emit(status.capitalize(), 0)

            self.progress_update.emit("Download complete!", 100)
            self.finished.emit(True)
        except Exception as e:
            self.progress_update.emit(f"Download error: {e}", 0)
            self.finished.emit(False)


class AIWorker(QObject):
    ai_result = Signal(str)
    status_update = Signal(str)
    finished = Signal()

    @Slot(str, str, str)
    def process_image(self, img_b64, prompt, model_name):
        try:
            self.status_update.emit(f"Analyzing with {model_name}...")
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.1
                }
            }
            res = requests.post(f"{OLLAMA_URL}/generate", json=payload, timeout=60).json()

            t_stamp = time.strftime("%H:%M:%S")
            if "error" in res:
                self.ai_result.emit(f'<span style="color:#da4453;"><b>[{t_stamp}] Error:</b> {res["error"]}</span><br><br>')
            else:
                raw_ans = res.get("response", "")
                ans = re.sub(r'<think>.*?</think>', '', raw_ans, flags=re.DOTALL).strip()
                ans = ans.replace('\n', '<br>')

                if not ans:
                    ans = "<i>[Empty response]</i>"
                self.ai_result.emit(f'<span style="color:#3daee9;"><b>[{t_stamp}] {model_name}:</b></span><br>{ans}<br><br>')
        except Exception as e:
            t_stamp = time.strftime("%H:%M:%S")
            self.ai_result.emit(f'<span style="color:#da4453;"><b>[{t_stamp}] Request error:</b> {e}</span><br><br>')
        finally:
            self.status_update.emit("Idle")
            self.finished.emit()


class AIScreenReader(QMainWindow):
    request_processing = Signal(str, str, str)
    request_pull = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Screen Reader")
        self.resize(500, 600)
        self.setWindowFlags(Qt.WindowStaysOnTopHint)

        self.setStyleSheet("""
            QMainWindow { background-color: #232629; }
            QLabel { color: #fcfcfc; font-size: 13px; }
            QPushButton { background-color: #3daee9; color: white; padding: 7px; border-radius: 4px; font-weight: bold; border: none; }
            QPushButton:hover { background-color: #1d99f3; }
            QPushButton:disabled { background-color: #4d4d4d; color: #888888; }
            QComboBox { padding: 6px; background-color: #31363b; color: white; border: 1px solid #4d4d4d; border-radius: 4px; }
            QTextEdit { background-color: #1a1c1f; color: #eff0f1; border: 1px solid #4d4d4d; border-radius: 4px; font-size: 14px; padding: 6px; }
            QFrame#card { background-color: #31363b; border-radius: 6px; border: 1px solid #4d4d4d; }
            QProgressBar { border: 1px solid #4d4d4d; border-radius: 4px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #3daee9; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(12)

        engine_bar = QHBoxLayout()
        self.engine_status_dot = QLabel("●")
        self.engine_status_dot.setStyleSheet("color: #da4453; font-size: 14px;")
        self.engine_status_label = QLabel("Checking Engine...")
        engine_bar.addWidget(self.engine_status_dot)
        engine_bar.addWidget(self.engine_status_label)
        engine_bar.addStretch()
        layout.addLayout(engine_bar)

        controls_layout = QHBoxLayout()
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(list(MODES.keys()))
        controls_layout.addWidget(self.mode_selector)

        self.model_selector = QComboBox()
        controls_layout.addWidget(self.model_selector)

        # New force scan button
        self.scan_btn = QPushButton("Scan Now")
        self.scan_btn.clicked.connect(self.force_scan)
        controls_layout.addWidget(self.scan_btn)

        layout.addLayout(controls_layout)

        output_header = QHBoxLayout()
        output_header.addWidget(QLabel("Live AI Log:"))
        self.ai_status_label = QLabel("Idle")
        self.ai_status_label.setStyleSheet("color: #aaaaaa; font-style: italic;")
        self.ai_status_label.setAlignment(Qt.AlignRight)
        output_header.addWidget(self.ai_status_label)
        layout.addLayout(output_header)

        self.ai_output = QTextEdit()
        self.ai_output.setReadOnly(True)
        self.ai_output.setMinimumHeight(150)
        layout.addWidget(self.ai_output)

        layout.addWidget(QLabel("<b>Model Catalog</b>"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        catalog_widget = QWidget()
        self.catalog_layout = QVBoxLayout(catalog_widget)
        self.catalog_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(catalog_widget)
        layout.addWidget(scroll)

        self.dl_status_label = QLabel("")
        self.dl_progress = QProgressBar()
        self.dl_progress.hide()
        layout.addWidget(self.dl_status_label)
        layout.addWidget(self.dl_progress)

        self.last_capture_time = 0.0
        self.req_start_time = 0.0
        self.busy = False
        self.pulling = False
        self.last_thumb_bytes = None
        self.engine_online = False

        self.overlay = SpectacleOverlay()
        self.overlay.first_selection_done.connect(self.on_first_selection)
        self.overlay.box_changed.connect(self.on_box_changed)
        self.overlay.selection_canceled.connect(self.on_selection_canceled)
        self.target_rect = None

        self.engine_timer = QTimer(self)
        self.engine_timer.timeout.connect(self.check_engine_status)
        self.engine_timer.start(3000)
        self.check_engine_status()

        self.ai_thread = QThread()
        self.ai_worker = AIWorker()
        self.ai_worker.moveToThread(self.ai_thread)
        self.request_processing.connect(self.ai_worker.process_image)
        self.ai_worker.ai_result.connect(self.append_log)
        self.ai_worker.status_update.connect(self.ai_status_label.setText)
        self.ai_worker.finished.connect(self.unlock_processing)
        self.ai_thread.start()

        self.dl_thread = QThread()
        self.dl_worker = ModelDownloader()
        self.dl_worker.moveToThread(self.dl_thread)
        self.request_pull.connect(self.dl_worker.pull)
        self.dl_worker.progress_update.connect(self.update_dl_progress)
        self.dl_worker.finished.connect(self.on_dl_finished)
        self.dl_thread.start()

        self.capture = QScreenCapture()
        self.session = QMediaCaptureSession()
        self.sink = QVideoSink()
        self.session.setScreenCapture(self.capture)
        self.session.setVideoOutput(self.sink)
        self.sink.videoFrameChanged.connect(self.handle_frame)

    @Slot()
    def force_scan(self):
        """Forces an immediate frame scan regardless of motion detection"""
        self.last_thumb_bytes = None
        self.last_capture_time = 0.0

    @Slot(str)
    def append_log(self, html_text):
        self.ai_output.append(html_text)
        sb = self.ai_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot()
    def check_engine_status(self):
        try:
            res = requests.get(f"{OLLAMA_URL}/tags", timeout=1)
            is_online = (res.status_code == 200)
        except:
            is_online = False

        if is_online != self.engine_online or self.model_selector.count() == 0:
            self.engine_online = is_online
            if is_online:
                self.engine_status_dot.setStyleSheet("color: #2ecc71; font-size: 14px;")
                self.engine_status_label.setText("Engine Online")
                self.refresh_catalog()
            else:
                self.engine_status_dot.setStyleSheet("color: #da4453; font-size: 14px;")
                self.engine_status_label.setText("Engine Offline (Start Ollama manually)")
                self.model_selector.clear()
                self.model_selector.addItem("Engine Offline")

    def start_flow(self):
        self.overlay.start_selection()

    @Slot(QRect)
    def on_first_selection(self, rect):
        self.target_rect = rect
        self.capture.setActive(True)

        screen = QApplication.primaryScreen().geometry()
        dock_x = rect.right() + 15
        dock_y = max(screen.top() + 10, min(rect.top(), screen.bottom() - self.height() - 10))
        if dock_x + self.width() > screen.right() - 10:
            dock_x = rect.left() - self.width() - 15
        if dock_x < screen.left() + 10:
            dock_x = screen.left() + 10

        self.move(dock_x, dock_y)

        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.show()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.show()
        self.raise_()
        self.activateWindow()

    @Slot(QRect)
    def on_box_changed(self, rect):
        self.target_rect = rect
        self.last_thumb_bytes = None

    @Slot()
    def on_selection_canceled(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def refresh_catalog(self):
        if not self.engine_online:
            return

        for i in reversed(range(self.catalog_layout.count())):
            widget = self.catalog_layout.itemAt(i).widget()
            if widget: widget.deleteLater()

        installed = []
        try:
            res = requests.get(f"{OLLAMA_URL}/tags", timeout=3)
            if res.status_code == 200:
                for m in res.json().get("models", []):
                    installed.append(m["name"])
        except: pass

        self.model_selector.clear()
        matching = [m for m in installed if any(v in m.lower() for v in [c['id'] for c in MODEL_CATALOG])]
        if matching:
            self.model_selector.addItems(matching)
        else:
            self.model_selector.addItem("No vision models installed")

        for model in MODEL_CATALOG:
            card = QFrame()
            card.setObjectName("card")
            c_layout = QGridLayout(card)

            title = QLabel(f"<b>{model['name']}</b> ({model['id']})")
            desc = QLabel(model['desc'])
            desc.setStyleSheet("color: #aaaaaa;")
            desc.setWordWrap(True)
            stats = QLabel(f"Speed: {model['speed']} | Accuracy: {model['accuracy']}")

            btn = QPushButton()

            if any(model['id'].lower() in inst.lower() for inst in installed):
                title.setStyleSheet("color: #3daee9;")
                btn.setText("Uninstall")
                btn.setStyleSheet("background-color: #da4453;")

                # Retrieve the exact tagged name (e.g. qwen3-vl:latest) for deletion
                exact_name = next(inst for inst in installed if model['id'].lower() in inst.lower())
                btn.clicked.connect(lambda checked=False, m_name=exact_name: self.uninstall_model(m_name))
                if self.pulling: btn.setEnabled(False)
            else:
                btn.setText("Download")
                btn.setStyleSheet("background-color: #3daee9;")
                btn.clicked.connect(lambda checked=False, m_id=model['id']: self.start_download(m_id))
                if self.pulling: btn.setEnabled(False)

            c_layout.addWidget(title, 0, 0)
            c_layout.addWidget(btn, 0, 1, 2, 1, Qt.AlignRight)
            c_layout.addWidget(stats, 1, 0)
            c_layout.addWidget(desc, 2, 0, 1, 2)
            self.catalog_layout.addWidget(card)

    def uninstall_model(self, model_name):
        if self.pulling: return
        try:
            requests.delete(f"{OLLAMA_URL}/delete", json={"name": model_name}, timeout=5)
            self.refresh_catalog()
        except:
            pass

    def start_download(self, model_id):
        if self.pulling: return
        self.pulling = True
        self.dl_progress.setValue(0)
        self.dl_progress.show()
        self.refresh_catalog()
        self.request_pull.emit(model_id)

    @Slot(str, int)
    def update_dl_progress(self, status, pct):
        self.dl_status_label.setText(status)
        self.dl_progress.setValue(pct)

    @Slot(bool)
    def on_dl_finished(self, success):
        self.pulling = False
        self.dl_progress.hide()
        if success: self.dl_status_label.setText("Installation Complete!")
        self.refresh_catalog()

    @Slot()
    def unlock_processing(self):
        self.busy = False

    @Slot(QVideoFrame)
    def handle_frame(self, frame):
        now = time.time()

        if self.busy:
            if now - self.req_start_time > 60:
                self.busy = False
            else:
                return

        if self.pulling or not self.engine_online or (now - self.last_capture_time < 1.5): return
        if not frame.isValid() or not self.target_rect: return

        image = frame.toImage()
        if image.isNull(): return

        screen = QApplication.primaryScreen().geometry()
        scale_x = image.width() / screen.width() if screen.width() > 0 else 1.0
        scale_y = image.height() / screen.height() if screen.height() > 0 else 1.0

        r = self.target_rect
        scaled_crop = QRect(
            int(r.x() * scale_x),
            int(r.y() * scale_y),
            int(r.width() * scale_x),
            int(r.height() * scale_y)
        )

        crop_rect = scaled_crop.intersected(image.rect())
        if crop_rect.isEmpty() or crop_rect.width() < 10 or crop_rect.height() < 10:
            return

        cropped = image.copy(crop_rect).convertToFormat(QImage.Format_RGB888)

        max_dim = 768
        if cropped.width() > max_dim or cropped.height() > max_dim:
            cropped = cropped.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Flawless Numpy Motion Detection
        thumb = cropped.scaled(32, 32, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).convertToFormat(QImage.Format_Grayscale8)
        ptr = thumb.constBits()
        # Direct byte parsing to array, dropping headers entirely
        current_thumb = np.frombuffer(ptr, dtype=np.uint8).copy()

        if self.last_thumb_bytes is not None and len(self.last_thumb_bytes) == len(current_thumb):
            diff = np.mean(np.abs(current_thumb.astype(int) - self.last_thumb_bytes.astype(int)))
            if diff < 2.0:
                return
        self.last_thumb_bytes = current_thumb
        self.last_capture_time = now

        png_ba = QByteArray()
        p_buf = QBuffer(png_ba)
        p_buf.open(QIODevice.WriteOnly)
        cropped.save(p_buf, "PNG")
        img_b64 = base64.b64encode(png_ba.data()).decode('utf-8')

        current_mode = self.mode_selector.currentText()
        prompt_template = MODES[current_mode]
        current_model = self.model_selector.currentText()

        if "No vision models installed" in current_model:
            return

        self.busy = True
        self.req_start_time = time.time()
        self.request_processing.emit(img_b64, prompt_template, current_model)

    def closeEvent(self, event):
        try:
            self.capture.setActive(False)
            self.overlay.close()
        except: pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = AIScreenReader()
    window.start_flow()
    sys.exit(app.exec())
