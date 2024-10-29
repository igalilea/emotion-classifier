import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def setup_processed_dirs(base_path: str) -> tuple:
    """
    Crea los directorios necesarios para las imágenes procesadas
    """
    # Crear directorio processed si no existe
    processed_path = Path(base_path) / 'processed'
    processed_train = processed_path / 'train'
    processed_test = processed_path / 'test'
    
    # Crear directorios si no existen
    for dir_path in [processed_path, processed_train, processed_test]:
        dir_path.mkdir(exist_ok=True)
        
    return processed_train, processed_test

def load_and_preprocess_images(input_dir: Path) -> tuple:
    """
    Carga y preprocesa todas las imágenes de un directorio
    """
    images = []
    labels = []
    filenames = []
    
    # Recorrer todas las imágenes en el directorio
    for img_path in tqdm(list(input_dir.glob('*.jpg'))):
        # Leer imagen
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Verificar tamaño
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))
            
            # Preprocesamiento básico
            img = img.astype(np.float32)
            img = img / 255.0  # Normalizar a [0,1]
            
            # Obtener etiqueta del nombre del archivo
            # Asumiendo que los archivos tienen formato: emotion_label_xxxx.jpg
            label = int(img_path.stem.split('_')[1])
            
            images.append(img)
            labels.append(label)
            filenames.append(img_path.name)
    
    return np.array(images), np.array(labels), filenames

def save_processed_data(images: np.array, labels: np.array, filenames: list, output_dir: Path):
    """
    Guarda las imágenes procesadas y sus etiquetas
    """
    # Guardar cada imagen procesada
    for img, label, filename in tqdm(zip(images, labels, filenames), total=len(images)):
        # Crear nombre de archivo procesado
        processed_filename = output_dir / filename
        
        # Guardar imagen
        cv2.imwrite(str(processed_filename), (img * 255).astype(np.uint8))
    
    # Guardar etiquetas en un archivo numpy
    labels_file = output_dir / 'labels.npy'
    np.save(str(labels_file), labels)

def main():
    # Configurar directorios
    base_path = Path('data')
    raw_train = base_path / 'raw' / 'train'
    raw_test = base_path / 'raw' / 'test'
    processed_train, processed_test = setup_processed_dirs(base_path)
    
    # Procesar conjunto de entrenamiento
    print("Procesando imágenes de entrenamiento...")
    train_images, train_labels, train_filenames = load_and_preprocess_images(raw_train)
    
    # Calcular estadísticas del conjunto de entrenamiento
    train_mean = train_images.mean()
    train_std = train_images.std()
    
    # Estandarizar conjunto de entrenamiento
    train_images = (train_images - train_mean) / train_std
    
    # Procesar conjunto de prueba
    print("\nProcesando imágenes de prueba...")
    test_images, test_labels, test_filenames = load_and_preprocess_images(raw_test)
    
    # Estandarizar conjunto de prueba usando estadísticas del entrenamiento
    test_images = (test_images - train_mean) / train_std
    
    # Guardar datos procesados
    print("\nGuardando imágenes procesadas...")
    save_processed_data(train_images, train_labels, train_filenames, processed_train)
    save_processed_data(test_images, test_labels, test_filenames, processed_test)
    
    # Guardar estadísticas de normalización
    stats = {
        'mean': train_mean,
        'std': train_std
    }
    np.save(str(Path(base_path) / 'processed' / 'normalization_stats.npy'), stats)
    
    # Imprimir resumen
    print("\nResumen del preprocesamiento:")
    print(f"Imágenes de entrenamiento procesadas: {len(train_images)}")
    print(f"Imágenes de prueba procesadas: {len(test_images)}")
    print(f"Media del conjunto de entrenamiento: {train_mean:.4f}")
    print(f"Desviación estándar del entrenamiento: {train_std:.4f}")

if __name__ == "__main__":
    main()