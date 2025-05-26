import os

def borrar_archivos_carpeta(carpeta):
    # Verifica si la carpeta existe
    if not os.path.exists(carpeta):
        print("La carpeta no existe.")
        return

    archivos = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, f))]

    if not archivos:
        print("No hay archivos para borrar.")
        return

    contador = 0
    for ruta_archivo in archivos:
        try:
            os.remove(ruta_archivo)  # Elimina el archivo
            contador += 1
            if contador % 100 == 0:  # Cada 100 archivos borrados
                print(f"{contador} archivos borrados.")
        except Exception as e:
            print(f"Error al borrar {ruta_archivo}: {e}")

    print(f"Se han borrado {contador} archivos en total.")

# Cambia esta ruta a la de tu carpeta
carpeta = "D:\RAE\dataset_es"
borrar_archivos_carpeta(carpeta)