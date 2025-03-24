import os

current_dir = os.path.dirname(__file__)  # Directorio actual de chat_simple.py
file_path = os.path.join(current_dir, '../pdf/GuiaViajeBariloche.pdf')

if os.path.exists(file_path):
    print("El archivo existe")
else:
    print("El archivo no existe")