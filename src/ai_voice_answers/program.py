#!/usr/bin/python3

# pip install PyQt5 numpy sounddevice pydub gTTS deep-consultation
# sudo apt install ffmpeg

import os
import sys
import time
import atexit
import signal
import shutil
import tempfile
import subprocess
import numpy as np
import sounddevice as sd

from pydub import AudioSegment

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QTextEdit, QFileDialog, 
    QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QSystemTrayIcon, QMenu, QAction, QSizePolicy, QSpacerItem, QCheckBox
)
from PyQt5.QtGui  import QIcon, QDesktopServices
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal

import ai_voice_answers.about             as about
import ai_voice_answers.modules.configure as configure 

from ai_voice_answers.modules.resources import resource_path
from ai_voice_answers.modules.wabout    import show_about_window
from ai_voice_answers.desktop import create_desktop_file
from ai_voice_answers.desktop import create_desktop_directory
from ai_voice_answers.desktop import create_desktop_menu

from deep_consultation.chat_deepinfra    import ChatDeepInfra
from ai_voice_answers.modules.consult    import transcription_in_depth
from ai_voice_answers.modules.work_audio import text_to_audio_file
from ai_voice_answers.modules.work_audio import play_audio_file

# ---------- Path to config file ----------
CONFIG_PATH = os.path.join( os.path.expanduser("~"),
                            ".config", 
                            about.__package__, 
                            "config.json" )

DEFAULT_CONTENT={   
    "menubar_show_recorder": "Show window recorder",
    "menubar_hide_recorder": "Hide window recorder",
    "menubar_configure": "✨ Configure window",
    "menubar_configure_gpt": "✨ Configure LLM",
    "menubar_about": "🌟 About",
    "menubar_coffee": "☕ Buy me a coffee",
    "menubar_exit": "❌ Exit",
    "windows_no_apikey": "No api_key in",
    "windows_loaded_apikey": "Loaded api_key",
    "windows_transcription": "Transcription",
    "windows_response_obtained": "Response obtained",
    "window_file_not_exist": "📁 File does not exist",
    "window_paying_audio": "📁 Playing audio",
    "window_success_saving": "Success saving audio",
    "window_error_saving": "Error saving audio",
    "window_done": "Done",
    "window_save_as": "Save as",
    "window_save_as_default": "resposta.mp3",
    "window_discard_audio": "Discard audio",
    "window_no_audio_record": "No audio captured",
    "window_audio_record": "Audio captured",
    "window_recording": "🎙️ Recording...",
    "window_recording_ended": "⏹️ Recording ended...",
    "window_use_history": "Use history",
    "window_use_history_tooltip": "Use history i chat list",
    "window_button_record": "Record",
    "window_button_record_tooltip": "Init audio record",
    "window_button_stop": "Stop record",
    "window_button_stop_tooltip": "Stop audio record",
    "window_button_stop_proc": "Stop record+processing",
    "window_button_stop_proc_tooltip": "Stop audio recording and start processing",
    "window_button_discard": "Discard recorded audio",
    "window_button_discard_tooltip": "Discard recorded audio",
    "window_button_play_recorded": "Play recorded audio",
    "window_button_play_recorded_tooltip": "Play recorded audio",
    "window_button_processing": "Processing",
    "window_button_processing_tooltip": "Processing the recorded audio",
    "window_button_ready": "Ready to record",
    "window_button_save_as": "Save as",
    "window_button_save_as_tooltip": "Save as the response audio",
    "window_button_play_response": "Play response audio",
    "window_button_play_response_tooltip": "Play response audio",
    "window_width": 600, 
    "window_height": 400
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
    "model_llm": "deepseek-ai/DeepSeek-V3.2",
    "model_transcript": "mistralai/Voxtral-Mini-3B-2507",
    "language": "pt",
    "play_factor": 1.5
}

configure.verify_default_config(CONFIG_GPT_PATH,default_content=DEFAULT_GPT_CONTENT)

################################################################################

def open_file_in_text_editor(filepath):
    if os.name == 'nt':  # Windows
        os.startfile(filepath)
    elif sys.platform == 'darwin':  # macOS
        subprocess.run(["open", filepath])
    elif os.name == 'posix':  # Linux/macOS
        subprocess.run(['xdg-open', filepath])

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
        print(CONFIG["window_recording"])

    def stop(self):
        if not self.recording:
            return None
        self.recording = False
        self.stream.stop()
        self.stream.close()
        self.stream = None
        print(CONFIG["window_recording_ended"])

        if not self.frames:
            return None

        return np.concatenate(self.frames, axis=0)


# =========================
# PROCESSING THREAD
# =========================
class ProcessingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)

    def __init__(self, audio_path, dir_temp, use_history=True, cdi=None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.dir_temp = dir_temp
        self.use_history = use_history
        # Se não veio um cdi, cria novo
        self.cdi = cdi

    def run(self):
        # progress
        self.progress.emit(0,"")
        
        config_gpt = configure.load_config(CONFIG_GPT_PATH)
        
        if len(config_gpt["api_key"].strip()) == 0:
            self.progress.emit(0,CONFIG["windows_no_apikey"]+": "+CONFIG_GPT_PATH)
            self.finished.emit({"error": "no_api_key"})
            return        

        self.progress.emit(5,CONFIG["windows_loaded_apikey"])
        print("📝 "+CONFIG["windows_loaded_apikey"])
        
        language = config_gpt["language"]
        
        transcription = transcription_in_depth(config_gpt, self.audio_path, language=language)
        
        # progress
        self.progress.emit(45,CONFIG["windows_transcription"]+":\n"+transcription)
        print("📝 "+CONFIG["windows_transcription"]+": " + transcription)
        
        if self.cdi is None:
            self.cdi = ChatDeepInfra(   config_gpt["base_url"], 
                                        config_gpt["api_key"], 
                                        config_gpt["model_llm"])
        
        # Sempre define o system prompt antes de perguntar
        SYSTEM_PROMPT = """
        You are an expert in many fields, a true guru. 
        Your mission is to respond to any question asked by the user. 
        If you do not know the answer, you must be honest and clearly state that you do not have it. 
        Your response should be a short, concise paragraph.
        Avoid trivial conversations, redundant answers, and idle chatter.
        Your personality is stoic and spartan.
        """
        self.cdi.set_system_prompt(SYSTEM_PROMPT)
        
        # Checa se histórico deve ser usado
        if self.use_history:
            res = self.cdi.chat(transcription)
        else:
            res = self.cdi.ask_once(transcription)
        
        # progress
        self.progress.emit(90,res)
        print("📝 "+CONFIG["windows_response_obtained"]+": ", res)
        
        res_audio_path = text_to_audio_file(res,language, self.dir_temp)
        
        # progress
        self.progress.emit(100,res)
        
        out = {
            "transcription" : transcription,
            "transcription_audio_path" : self.audio_path,
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

        self.cdi = None

        self.temp_dir = tempfile.mkdtemp(prefix=about.__package__+"_")
        atexit.register(self.cleanup_temp_dir)
        
        self.setWindowTitle(about.__program_name__)
        self.setGeometry(200, 200, CONFIG["window_width"], CONFIG["window_height"])
        
        ## Icon
        # Get base directory for icons
        self.icon_path = resource_path('icons', 'logo.png')
        self.setWindowIcon(QIcon(self.icon_path))         

        self.recorder = AudioRecorder()
        self.audio_data = None
        self.audio_path = None
        self.audio_res_path = None

        self._build_ui()

    def cleanup_temp_dir(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()
        
        # Checkbox para ativar ou desativar histórico/memory
        self.use_history_checkbox = QCheckBox(CONFIG["window_use_history"])
        self.use_history_checkbox.setToolTip(CONFIG["window_button_record_tooltip"])
        self.use_history_checkbox.setChecked(True)  # default ON
        layout.addWidget(self.use_history_checkbox)
        
        self.record_btn = QPushButton(CONFIG["window_button_record"])
        self.record_btn.setIcon(QIcon.fromTheme("media-record"))
        self.record_btn.setToolTip(CONFIG["window_button_record_tooltip"])
        self.record_btn.clicked.connect(self.start_recording)
        layout.addWidget(self.record_btn)

        # Layout horizontal para stop
        buttons_stop_layout = QHBoxLayout()
        
        self.stop_btn = QPushButton(CONFIG["window_button_stop"])
        self.stop_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_btn.setToolTip(CONFIG["window_button_stop_tooltip"])
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        buttons_stop_layout.addWidget(self.stop_btn)
        
        self.stop_proc_btn = QPushButton(CONFIG["window_button_stop_proc"])
        self.stop_proc_btn.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.stop_proc_btn.setToolTip(CONFIG["window_button_stop_proc_tooltip"])
        self.stop_proc_btn.setEnabled(False)
        self.stop_proc_btn.clicked.connect(self.stop_recording_and_proc)
        buttons_stop_layout.addWidget(self.stop_proc_btn)
        
        # Adicionar layout horizontal ao layout principal
        layout.addLayout(buttons_stop_layout)
        
        # Layout horizontal para descartar + play
        buttons_layout = QHBoxLayout()

        self.discard_btn = QPushButton(CONFIG["window_button_discard"])
        self.discard_btn.setIcon(QIcon.fromTheme("user-trash"))
        self.discard_btn.setToolTip(CONFIG["window_button_discard_tooltip"])
        self.discard_btn.setEnabled(False)
        self.discard_btn.clicked.connect(self.discard_input_audio)
        buttons_layout.addWidget(self.discard_btn)

        self.play_btn = QPushButton(CONFIG["window_button_play_recorded"])
        self.play_btn.setIcon(QIcon.fromTheme("multimedia-player"))
        self.play_btn.setToolTip(CONFIG["window_button_play_recorded_tooltip"])
        self.play_btn.setEnabled(False)  # Pode habilitar quando tiver áudio
        self.play_btn.clicked.connect(self.play_input_audio)
        buttons_layout.addWidget(self.play_btn)

        # Adicionar layout horizontal ao layout principal
        layout.addLayout(buttons_layout)
        
        # Criar espaçamento vertical de 20 pixels
        layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        self.process_btn = QPushButton(CONFIG["window_button_processing"])
        self.process_btn.setIcon(QIcon.fromTheme("system-run"))
        self.process_btn.setToolTip(CONFIG["window_button_processing_tooltip"])
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_audio)
        layout.addWidget(self.process_btn)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText(CONFIG["window_button_ready"])
        self.status_text.setWordWrapMode(True)
        self.status_text.setFrameStyle(QFrame.NoFrame)
        self.status_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.status_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.status_text)

        # Layout horizontal para Save as + play
        buttons_res_layout = QHBoxLayout()

        self.save_as_btn = QPushButton(CONFIG["window_button_save_as"])
        self.save_as_btn.setIcon(QIcon.fromTheme("document-save-as"))
        self.save_as_btn.setToolTip(CONFIG["window_button_save_as_tooltip"])
        self.save_as_btn.setEnabled(False)
        self.save_as_btn.clicked.connect(self.save_as_res_audio)
        buttons_res_layout.addWidget(self.save_as_btn)

        self.play_res_btn = QPushButton(CONFIG["window_button_play_response"])
        self.play_res_btn.setIcon(QIcon.fromTheme("multimedia-player"))
        self.play_res_btn.setToolTip(CONFIG["window_button_play_response_tooltip"])
        self.play_res_btn.setEnabled(False)  # Pode habilitar quando tiver áudio
        self.play_res_btn.clicked.connect(self.play_res_audio)
        buttons_res_layout.addWidget(self.play_res_btn)

        # Adicionar layout horizontal ao layout principal
        layout.addLayout(buttons_res_layout)


        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        central.setLayout(layout)

    # -------------------------
    # RECORD CONTROL
    # -------------------------
    def start_recording(self):
        self.recorder.start()
        self.status_text.setText(CONFIG["window_recording"])
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_proc_btn.setEnabled(True)

    def stop_recording(self):
        audio = self.recorder.stop()
        self.stop_btn.setEnabled(False)
        self.stop_proc_btn.setEnabled(False)
        self.record_btn.setEnabled(True)

        if audio is not None:
            self.audio_data = audio
            self.audio_path = self.save_input_audio_mp3(audio)
            self.status_text.setText(CONFIG["window_audio_record"])
            self.process_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.discard_btn.setEnabled(True)
        else:
            self.status_text.setText(CONFIG["window_no_audio_record"])

    def stop_recording_and_proc(self):
        self.stop_recording()
        self.process_audio()

    def save_input_audio_mp3(self, audio_data):
        audio_int16 = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)

        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=self.recorder.samplerate,
            sample_width=2,
            channels=1
        )

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp3",
            delete=False,
            dir=self.temp_dir
        )
        tmp.close()

        audio_segment.export(tmp.name, format="mp3")
        return tmp.name
        
    # -------------------------
    # ACTIONS
    # -------------------------
    def discard_input_audio(self):
        self.audio_data = None
        if self.audio_path and os.path.isfile(self.audio_path):
            os.remove(self.audio_path)
        self.audio_path = None
        self.process_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.status_text.setText(CONFIG["window_discard_audio"])
    
    def process_audio(self):
        if self.audio_data is None or self.audio_path is None:
            return

        self.process_btn.setEnabled(False)

        self.progress.setVisible(True)
        self.progress_callback(0,"")
        
        if self.audio_res_path and os.path.isfile(self.audio_res_path):
            os.remove(self.audio_res_path)

        self.worker = ProcessingThread(
            self.audio_path,
            self.temp_dir,
            use_history = self.use_history_checkbox.isChecked(),
            cdi = self.cdi
        )
        
        self.worker.progress.connect(self.progress_callback)
        self.worker.finished.connect(self.processing_done)
        self.worker.start()

    def progress_callback(self, value, msg):
        self.progress.setValue(value)
        self.status_text.setText(msg)

    def play_input_audio(self):
        if os.path.isfile(self.audio_path):
            print(CONFIG["window_paying_audio"]+": "+self.audio_path)
            
            if hasattr(self, "player") and self.player.isRunning():
                return  # ou pare a anterior
            
            self.player = AudioPlayerThread(self.audio_path, fator=1.0)
            self.player.finished.connect(
                lambda: self.statusBar().showMessage(CONFIG["window_done"], 3000)
            )
            self.player.start()
        else:
            print(CONFIG["window_file_not_exist"]+": "+self.audio_path)
    
    def processing_done(self, data):
            
        if not data:
            #self.status_text.setText("Erro no processamento")
            self.progress.setVisible(False)
            return

        if "error" in data and data["error"] == "no_api_key":
            open_file_in_text_editor(CONFIG_GPT_PATH)
            self.progress.setVisible(False)
            return
        
        # Atualiza o cdi do MainWindow para manter histórico
        if hasattr(self.worker, "cdi") and self.worker.cdi is not None:
            self.cdi = self.worker.cdi
        
        self.status_text.setText(data["response"])
        self.audio_res_path = data["response_audio_path"]

        self.progress.setVisible(False)
        
        self.play_res_audio()
        
        self.save_as_btn.setEnabled(True)
        self.play_res_btn.setEnabled(True)
            
    def save_as_res_audio(self):
        if not self.audio_res_path or not os.path.isfile(self.audio_res_path):
            return

        # Nome sugerido
        default_name = CONFIG["window_save_as_default"]

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            CONFIG["window_save_as"],
            default_name,
            "Audio Files (*.mp3)"
        )

        if not file_path:
            return  # usuário cancelou

        # garantir extensão
        if not file_path.lower().endswith(".mp3"):
            file_path += ".mp3"

        try:
            shutil.copyfile(self.audio_res_path, file_path)
            self.statusBar().showMessage(CONFIG["window_success_saving"], 4000)
        except Exception as e:
            self.statusBar().showMessage(CONFIG["window_error_saving"]+f": {e}", 6000)
        
    def play_res_audio(self):
        if os.path.isfile(self.audio_res_path):
            print(CONFIG["window_paying_audio"]+": "+self.audio_res_path)
            
            if hasattr(self, "player") and self.player.isRunning():
                return  # ou pare a anterior
            
            config_gpt = configure.load_config(CONFIG_GPT_PATH)
            
            self.player = AudioPlayerThread(self.audio_res_path, fator=config_gpt["play_factor"])
            self.player.finished.connect(
                lambda: self.statusBar().showMessage(CONFIG["window_done"], 3000)
            )
            self.player.start()
        else:
            msg = CONFIG["window_file_not_exist"]+": "+self.audio_res_path
            self.statusBar().showMessage(msg, 3000)
            print(msg)
            
    def closeEvent(self, event):
        event.ignore()
        self.hide()


# =========================
# SYSTEM TRAY INDICATOR
# =========================
class TrayIcon(QSystemTrayIcon):
    def __init__(self, window, app):
        super().__init__()

        ## Icon
        # Get base directory for icons
        self.icon_path = resource_path('icons', 'logo.png')

        self.window = window
        self.app = app

        self.setIcon(QIcon(self.icon_path))

        menu = QMenu()

        #
        show_action = QAction(  QIcon.fromTheme("view-fullscreen"), 
                                CONFIG["menubar_show_recorder"], 
                                self)
        show_action.triggered.connect(self.show_window)
        menu.addAction(show_action)

        #
        hide_action = QAction(  QIcon.fromTheme("view-restore"), 
                                CONFIG["menubar_hide_recorder"], 
                                self)
        hide_action.triggered.connect(self.hide_window)
        menu.addAction(hide_action)

        #
        menu.addSeparator()

        #
        self.configure_action = QAction(QIcon.fromTheme("document-properties"), 
                                        CONFIG["menubar_configure"], 
                                        self)
        self.configure_action.triggered.connect(self.open_configure_editor)
        menu.addAction(self.configure_action)
        
        #
        self.configure_gpt_action = QAction(QIcon.fromTheme("document-properties"), 
                                            CONFIG["menubar_configure_gpt"], 
                                            self)
        self.configure_gpt_action.triggered.connect(self.open_configure_gpt_editor)
        menu.addAction(self.configure_gpt_action)
        
        #
        self.about_action = QAction(QIcon.fromTheme("help-about"), 
                                    CONFIG["menubar_about"], 
                                    self)
        self.about_action.triggered.connect(self.open_about)
        menu.addAction(self.about_action)
        
        # Coffee
        self.coffee_action = QAction(   QIcon.fromTheme("emblem-favorite"), 
                                        CONFIG["menubar_coffee"], 
                                        self)
        self.coffee_action.triggered.connect(self.on_coffee_action_click)
        menu.addAction(self.coffee_action)
        
        #
        menu.addSeparator()
        
        #
        quit_action = QAction(CONFIG["menubar_exit"], self)
        quit_action.triggered.connect(self.quit_app)
        menu.addAction(quit_action)

        self.setContextMenu(menu)

        self.activated.connect(self.on_click)

    def open_configure_editor(self):
        open_file_in_text_editor(CONFIG_PATH)
        
    def open_configure_gpt_editor(self):
        open_file_in_text_editor(CONFIG_GPT_PATH)
    
    def open_about(self):
        data={
            "version": about.__version__,
            "package": about.__package__,
            "program_name": about.__program_name__,
            "author": about.__author__,
            "email": about.__email__,
            "description": about.__description__,
            "url_source": about.__url_source__,
            "url_doc": about.__url_doc__,
            "url_funding": about.__url_funding__,
            "url_bugs": about.__url_bugs__
        }
        show_about_window(data,self.icon_path)

    def on_coffee_action_click(self):
        QDesktopServices.openUrl(QUrl("https://ko-fi.com/trucomanx"))

    def show_window(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def hide_window(self):
        self.window.hide()
       
    def quit_app(self):
    
        # parar player se existir
        if hasattr(self.window, "player") and self.window.player.isRunning():
            self.window.player.quit()
            self.window.player.wait()

        # parar worker se existir
        if hasattr(self.window, "worker") and self.window.worker.isRunning():
            self.window.worker.quit()
            self.window.worker.wait()
    
        self.window.cleanup_temp_dir()
        self.hide()
        self.app.quit()

    def on_click(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.show_window()


# =========================
# MAIN
# =========================
def main():
    # Captura de sinal Ctrl+C no terminal
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    create_desktop_directory()    
    create_desktop_menu()
    create_desktop_file(os.path.join("~",".local","share","applications"))
    
    for n in range(len(sys.argv)):
        if sys.argv[n] == "--autostart":
            create_desktop_directory(overwrite = True)
            create_desktop_menu(overwrite = True)
            create_desktop_file(os.path.join("~",".config","autostart"), overwrite=True)
            return
        if sys.argv[n] == "--applications":
            create_desktop_directory(overwrite = True)
            create_desktop_menu(overwrite = True)
            create_desktop_file(os.path.join("~",".local","share","applications"), overwrite=True)
    
    app = QApplication(sys.argv)
    app.setApplicationName(about.__package__) # xprop WM_CLASS # *.desktop -> StartupWMClass  
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    window.hide()  # janela NÃO é principal

    tray = TrayIcon(window, app)
    tray.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

