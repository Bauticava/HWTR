import os

folder = "D:/RAE/dataset15_es"
count = 0  # Contador de archivos renombrados
deleted = 0  # Contador de archivos eliminados

for filename in os.listdir(folder):
    new_filename = filename.replace("ñ", "n")  # Cambia "ñ" por "n"

    if filename != new_filename:
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_filename)

        # Si ya existe un archivo con el nuevo nombre, omitimos el cambio y eliminamos el original
        if os.path.exists(new_path):
            os.remove(old_path)  # Borra el archivo original
            deleted += 1
            print(f"🗑️ Eliminado archivo duplicado: {old_path}")
        else:
            os.rename(old_path, new_path)  # Renombrar correctamente
            count += 1

            # Mostrar mensaje cada 20 cambios
            if count % 20 == 0:
                print(f"✅ {count} archivos renombrados hasta ahora...")

print(f"🎉 Proceso completado. Total de archivos renombrados: {count}")
print(f"🗑️ Total de archivos eliminados: {deleted}")