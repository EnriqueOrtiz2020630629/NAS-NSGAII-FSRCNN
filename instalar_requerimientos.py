import subprocess
import sys
import os
import requests

paquetes_requeridos = [
    "torch",
    "numpy",
    "matplotlib",
    "pymoo",
    "tqdm"
]

def instalar(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def checar_e_instalar():
    for package in paquetes_requeridos:
        try:
            __import__(package)
            print(f"{package} ya esta instalado.")
        except ImportError:
            print(f"{package} no esta instalado. Instalando...")
            instalar(package)

if __name__ == "__main__":
    checar_e_instalar()
   


    