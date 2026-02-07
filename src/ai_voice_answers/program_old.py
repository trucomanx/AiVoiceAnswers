import os

import {MODULE_NAME}.about as about
import {MODULE_NAME}.modules.configure as configure 
from {MODULE_NAME}.modules.resources import resource_path

# Caminho para o arquivo de configuração
CONFIG_PATH = os.path.join(os.path.expanduser("~"),".config",about.__package__,"config.json")

configure.verify_default_config(CONFIG_PATH, default_content={"casa":"verde"})

CONFIG=configure.load_config(CONFIG_PATH)

icon_path = resource_path('icons', 'logo.png')

def main():
    print("icon_path:",icon_path)
    print("Hola!")

if __name__ == "__main__":
    main()
