#!/usr/bin/python3

'''
## ubuntu ##
python3 -m venv venv-temporal
source venv-temporal/bin/activate
pip install --upgrade pip

## windows ##
python -m venv venv-temporal
venv-temporal\Scripts\activate
python -m pip install --upgrade pip


pip install pyinstaller pyinstaller-hooks-contrib
pip install -r requirements.txt
cd src


## ubuntu ##
python3 -m PyInstaller --onefile --windowed --name ai_voice_answers --add-data "ai_voice_answers/icons:icons" --collect-all PyQt5  program_launcher.py

## windows ##
python -m PyInstaller --onefile --windowed --name ai_voice_answers --add-data "ai_voice_answers/icons;icons" --collect-all PyQt5  program_launcher.py

'''

from ai_voice_answers.program import main

if __name__ == "__main__":
    main()

