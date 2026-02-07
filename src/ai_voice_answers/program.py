#!/usr/bin/python3

# pip install pyqt5 sounddevice numpy pydub
# sudo apt install ffmpeg

import os
import sys
import time
import tempfile
import numpy as np
import sounddevice as sd

from pydub import AudioSegment

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QTextEdit, 
    QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QSystemTrayIcon, QMenu, QAction, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import ai_voice_answers.about             as about
import ai_voice_answers.modules.configure as configure 

from ai_voice_answers.modules.resources import resource_path
from ai_voice_answers.modules.wabout    import show_about_window
from ai_voice_answers.desktop import create_desktop_file
from ai_voice_answers.desktop import create_desktop_directory
from ai_voice_answers.desktop import create_desktop_menu

from ai_voice_answers.modules.consult    import transcription_in_depth
from ai_voice_answers.modules.consult    import consultation_in_depth
from ai_voice_answers.modules.work_audio import text_to_audio_file
from ai_voice_answers.modules.work_audio import play_audio_file

# ---------- Path to config file ----------
CONFIG_PATH = os.path.join( os.path.expanduser("~"),
                            ".config", 
                            about.__package__, 
                            "config.json" )

DEFAULT_CONTENT={   
    "toolbar_configure": "Configure",
    "toolbar_configure_tooltip": "Open the configure Json file of program GUI",
    "toolbar_about": "About",
    "toolbar_about_tooltip": "About the program",
    "toolbar_coffee": "Coffee",
    "toolbar_coffee_tooltip": "Buy me a coffee (TrucomanX)",
    "window_width": 1024,
    "window_height": 800
}

configure.verify_default_config(CONFIG_PATH,default_content=DEFAULT_CONTENT)

CONFIG=configure.load_config(CONFIG_PATH)

# ---------- Path to config gpt file ----------
CONFIG_GPT_PATH = os.path.join( os.path.expanduser("~"),
                                ".config", 
                                about.__package__, 
                                "config.gpt.json" )

DEFAULT_GPT_CONTENT={
    "api_key": "",
    "usage": "https://deepinfra.com/dash/usage",
    "base_url": "https://api.deepinfra.com/v1/openai",
    "model_llm": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "model_transcript": "mistralai/Voxtral-Mini-3B-2507"
}

configure.verify_default_config(CONFIG_GPT_PATH,default_content=DEFAULT_GPT_CONTENT)



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
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)

    def __init__(self, audio_data, samplerate, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.samplerate = samplerate

    def run(self):
        # progress
        self.progress.emit(0,"")

        audio_int16 = np.clip(self.audio_data, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        
        # progress
        self.progress.emit(10,"Audio loaded in memory")

        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.samplerate,
            sample_width=2,
            channels=1
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()

        audio_segment.export(tmp.name, format="mp3")
        
        config_gpt = configure.load_config(CONFIG_GPT_PATH)
        
        if len(config_gpt["api_key"].strip()) == 0:
            self.progress.emit(0,"No api_key in:"+CONFIG_GPT_PATH)
            return        

        # progress
        self.progress.emit(30,"💾 MP3 saved")
        print("💾 MP3 salvo em:", tmp.name)
        
        language = "pt"
        fator    = 1.5
        
        transcription = transcription_in_depth(config_gpt, tmp.name, language=language)
        
        # progress
        self.progress.emit(60,"transcription:\n"+transcription)
        print("📝 transcription:", transcription)
        
        res = consultation_in_depth(config_gpt, transcription)
        
        # progress
        self.progress.emit(90,"response obtained")
        print("📝 result:", res)
        
        res_audio_path = text_to_audio_file(res,language)
        
        # progress
        self.progress.emit(100,"")
        
        # play_audio_file(res_audio_path, fator)
        out = {
            "transcription" : transcription,
            "transcription_audio_path" : tmp.name,
            "response": res,
            "response_audio_path": res_audio_path
        }
        self.finished.emit(out)

# =========================
# PLAY AUDIO THREAD
# =========================
class AudioPlayerThread(QThread):
    finished = pyqtSignal()

    def __init__(self, audio_path, fator=1.0, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.fator = fator
        self._running = True

    def run(self):
        if self._running:
            play_audio_file(self.audio_path, self.fator)
        self.finished.emit()

    def stop(self):
        self._running = False
        # se play_audio_file suportar stop → chamar aqui

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
        self.audio_path = None

        self._build_ui()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()
        
        self.record_btn = QPushButton("▶️ Gravar")
        self.record_btn.clicked.connect(self.start_recording)
        layout.addWidget(self.record_btn)

        self.stop_btn = QPushButton("⏹️ Finalizar gravação")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        layout.addWidget(self.stop_btn)
        
        # Layout horizontal para descartar + play
        buttons_layout = QHBoxLayout()

        self.discard_btn = QPushButton("Descartar")
        self.discard_btn.setEnabled(False)
        self.discard_btn.clicked.connect(self.discard_audio)
        buttons_layout.addWidget(self.discard_btn)

        self.play_btn = QPushButton("▶️ Play Audio")
        self.play_btn.setEnabled(False)  # Pode habilitar quando tiver áudio
        self.play_btn.clicked.connect(self.play_input_audio)
        buttons_layout.addWidget(self.play_btn)

        # Adicionar layout horizontal ao layout principal
        layout.addLayout(buttons_layout)
        
        # Criar espaçamento vertical de 20 pixels
        layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.process_btn = QPushButton("Processar")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_audio)
        layout.addWidget(self.process_btn)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Pronto para gravar")
        self.status_text.setWordWrapMode(True)

        # aparência de label
        self.status_text.setFrameStyle(QFrame.NoFrame)
        self.status_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.status_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        layout.addWidget(self.status_text)


        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        central.setLayout(layout)

    # -------------------------
    # RECORD CONTROL
    # -------------------------
    def start_recording(self):
        self.recorder.start()
        self.status_text.setText("🎙️ Gravando...")
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_recording(self):
        audio = self.recorder.stop()
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(True)

        if audio is not None:
            self.audio_data = audio
            self.audio_path = self.save_input_audio_mp3(audio)
            self.status_text.setText("Áudio gravado")
            self.process_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.discard_btn.setEnabled(True)
        else:
            self.status_text.setText("Nenhum áudio capturado")

    def save_input_audio_mp3(self, audio_data):
        audio_int16 = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)

        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.recorder.samplerate,
            sample_width=2,
            channels=1
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()

        audio_segment.export(tmp.name, format="mp3")
        return tmp.name
        
    # -------------------------
    # ACTIONS
    # -------------------------
    def discard_audio(self):
        self.audio_data = None
        os.remove(self.audio_path)
        self.audio_path = None
        self.process_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.status_text.setText("Áudio descartado")
        

    def process_audio(self):
        if self.audio_data is None or self.audio_path is None:
            return

        self.process_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)

        self.progress.setVisible(True)
        self.progress_callback(0,"")

        self.worker = ProcessingThread(
            self.audio_data.copy(),
            self.recorder.samplerate
        )
        self.worker.progress.connect(self.progress_callback)
        self.worker.finished.connect(self.processing_done)
        self.worker.start()

    def progress_callback(self, value, msg):
        self.progress.setValue(value)
        self.status_text.setText(msg)

    def play_input_audio(self):
        print("play_input_audio")
    
    def processing_done(self, data):
        self.status_text.setText(data["response"])
       
        self.progress.setVisible(False)
        
        print("📁 Arquivo final:", data["response_audio_path"])
        
        self.player = AudioPlayerThread(data["response_audio_path"], fator=1.5)
        self.player.finished.connect(
            lambda: self.statusBar().showMessage("Pronto", 3000)
        )
        self.player.start()

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


