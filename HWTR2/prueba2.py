import os
import csv

# Rutas de los archivos
base_path = "D:/RAE/"
dataset_path = os.path.join(base_path, "dataset15_es")
csv_file = os.path.join(dataset_path, "etiquetas.csv")
csv_temp = os.path.join(dataset_path, "etiquetas_temp.csv")  # Archivo temporal

count = 0  # Contador de rutas corregidas
skipped = 0  # Contador de rutas omitidas por duplicado
existing_paths = set()  # Para evitar duplicados

with open(csv_file, mode="r", encoding="utf-8-sig", newline="") as file_in, \
     open(csv_temp, mode="w", encoding="utf-8-sig", newline="") as file_out:

    reader = csv.reader(file_in)
    writer = csv.writer(file_out)

    for row in reader:
        old_image_path = row[0]  # Ruta de la imagen en el CSV
        label = row[1]           # Etiqueta de la palabra (NO se cambia)

        # Modificar la ruta si contiene "ñ"
        new_image_path = old_image_path.replace("ñ", "n")

        # Verificar si la ruta ya fue escrita antes
        if new_image_path in existing_paths:
            skipped += 1
            print(f"⚠️ Ruta duplicada omitida: {new_image_path}")
            continue  # Omitir esta línea para evitar duplicados

        existing_paths.add(new_image_path)  # Agregar a los existentes
        writer.writerow([new_image_path, label])  # Guardar la línea corregida
        count += 1

        # Mostrar mensaje cada 20 cambios
        if count % 20 == 0:
            print(f"✅ {count} rutas corregidas en el CSV hasta ahora...")

# Reemplazar el archivo original con el modificado
os.replace(csv_temp, csv_file)

print(f"🎉 Proceso completado. Total de rutas corregidas en el CSV: {count}")
print(f"⚠️ Total de rutas duplicadas omitidas: {skipped}")