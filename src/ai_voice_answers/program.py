#!/usr/bin/python3

# pip install pyqt5 sounddevice numpy pydub
# sudo apt install ffmpeg

import sys
import time
import tempfile
import numpy as np
import sounddevice as sd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QVBoxLayout, QLabel, QProgressBar,
    QSystemTrayIcon, QMenu, QAction
)
from PyQt5.QtGui import QIcon

from pydub import AudioSegment


# =========================
# AUDIO RECORDER
# =========================
class AudioRecorder:
    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.frames = []
        self.recording = False
        self.stream = None

    def _callback(self, indata, frames, time_, status):
        if self.recording:
            self.frames.append(indata.copy())

    def start(self):
        if self.recording:
            return
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback
        )
        self.stream.start()
        print("🎙️ Gravando...")

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.stream = None
        print("⏹️ Gravação finalizada")

        if not self.frames:
            return None

        return np.concatenate(self.frames, axis=0)


# =========================
# PROCESSING THREAD
# =========================
class ProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, audio_data, samplerate, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.samplerate = samplerate

    def run(self):
        for i in range(30):
            time.sleep(0.03)
            self.progress.emit(i)

        audio_int16 = np.clip(self.audio_data, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)

        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.samplerate,
            sample_width=2,
            channels=1
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()

        audio_segment.export(tmp.name, format="mp3")

        for i in range(30, 101):
            time.sleep(0.01)
            self.progress.emit(i)

        print("💾 MP3 salvo em:", tmp.name)
        self.finished.emit(tmp.name)


# =========================
# MAIN WINDOW
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Gravador de Áudio")
        self.setGeometry(200, 200, 420, 320)

        self.recorder = AudioRecorder()
        self.audio_data = None

        self._build_ui()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()

        self.status_label = QLabel("Pronto para gravar")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.record_btn = QPushButton("▶️ Gravar")
        self.record_btn.clicked.connect(self.start_recording)

        self.stop_btn = QPushButton("⏹️ Finalizar gravação")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)

        self.process_btn = QPushButton("Processar")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_audio)

        self.discard_btn = QPushButton("Descartar")
        self.discard_btn.setEnabled(False)
        self.discard_btn.clicked.connect(self.discard_audio)

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        layout.addWidget(self.status_label)
        layout.addWidget(self.record_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.discard_btn)
        layout.addWidget(self.progress)

        central.setLayout(layout)

    # -------------------------
    # RECORD CONTROL
    # -------------------------
    def start_recording(self):
        self.recorder.start()
        self.status_label.setText("🎙️ Gravando...")
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_recording(self):
        audio = self.recorder.stop()
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(True)

        if audio is not None:
            self.audio_data = audio
            self.status_label.setText("Áudio gravado")
            self.process_btn.setEnabled(True)
            self.discard_btn.setEnabled(True)
        else:
            self.status_label.setText("Nenhum áudio capturado")

    # -------------------------
    # ACTIONS
    # -------------------------
    def discard_audio(self):
        self.audio_data = None
        self.process_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.status_label.setText("Áudio descartado")

    def process_audio(self):
        if self.audio_data is None:
            return

        self.process_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)

        self.progress.setVisible(True)
        self.progress.setValue(0)

        self.worker = ProcessingThread(
            self.audio_data.copy(),
            self.recorder.samplerate
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.processing_done)
        self.worker.start()

    def processing_done(self, path):
        self.status_label.setText("MP3 salvo com sucesso")
        self.progress.setVisible(False)
        print("📁 Arquivo final:", path)

    def closeEvent(self, event):
        event.ignore()
        self.hide()


# =========================
# SYSTEM TRAY INDICATOR
# =========================
class TrayIcon(QSystemTrayIcon):
    def __init__(self, window, app):
        super().__init__()

        self.window = window
        self.app = app

        self.setIcon(QIcon.fromTheme("audio-input-microphone"))
        self.setToolTip("Gravador de Áudio")

        menu = QMenu()

        show_action = QAction("Abrir gravador", self)
        show_action.triggered.connect(self.show_window)

        hide_action = QAction("Ocultar janela", self)
        hide_action.triggered.connect(self.hide_window)

        quit_action = QAction("Sair", self)
        quit_action.triggered.connect(self.quit_app)

        menu.addAction(show_action)
        menu.addAction(hide_action)
        menu.addSeparator()
        menu.addAction(quit_action)

        self.setContextMenu(menu)

        self.activated.connect(self.on_click)

    def show_window(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def hide_window(self):
        self.window.hide()

    def quit_app(self):
        self.hide()
        self.app.quit()

    def on_click(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.show_window()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    window.hide()  # janela NÃO é principal

    tray = TrayIcon(window, app)
    tray.show()

    sys.exit(app.exec_())


