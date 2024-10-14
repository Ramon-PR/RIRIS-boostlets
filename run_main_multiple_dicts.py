import os
import subprocess
import hydra
from omegaconf import DictConfig

# Run like:
# python .\run_main_multiple_dicts.py +folder=saved_dicts/tanh_dicts

def obtener_archivos_mat(folder):
    archivos_mat = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".mat"):
                # Guardar el path relativo a 'folder'
                relative_path = os.path.relpath(os.path.join(root, file), folder)
                archivos_mat.append(relative_path)
    return archivos_mat

# Hydra para la configuración
@hydra.main(version_base=None, config_path=None)
def run_main(cfg: DictConfig):
    # folder = cfg.folder  # Carpeta base pasada como argumento a Hydra
    folder = cfg.get('folder', './saved_dicts')  # Si no se pasa 'folder', se usa './saved_dicts'
    archivos_mat = obtener_archivos_mat(folder)  # Buscar los archivos .mat

    # Ejecutar main.py para cada archivo .mat
    for archivo in archivos_mat:
        print(f"Ejecutando main.py con archivo: {archivo}")
        subprocess.run(["python", "main.py", f"folder_dict={folder}", f"file_dict={archivo}"])

if __name__ == "__main__":
    run_main()