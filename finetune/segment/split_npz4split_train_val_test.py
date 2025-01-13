import os
import numpy as np
from sklearn.model_selection import train_test_split

def resize_image_array(image_array, target_shape):
    """
    Ajusta o formato do array de imagem para o shape desejado.

    Args:
        image_array (np.ndarray): Array de imagem original.
        target_shape (tuple): Shape desejado (C, H, W).

    Returns:
        np.ndarray: Array ajustado.
    """
    from skimage.transform import resize
    resized = np.zeros(target_shape, dtype=image_array.dtype)
    for c in range(target_shape[0]):
        resized[c] = resize(
            image_array[c] if c < image_array.shape[0] else np.zeros_like(image_array[0]),
            target_shape[1:],
            mode='constant',
            preserve_range=True
        )
    return resized

def resize_label_array(label_array, target_shape):
    """
    Ajusta o formato do array de rótulo para o shape desejado.

    Args:
        label_array (np.ndarray): Array de rótulo original.
        target_shape (tuple): Shape desejado (1, H, W).

    Returns:
        np.ndarray: Array ajustado.
    """
    from skimage.transform import resize
    resized = resize(
        label_array[0],
        target_shape[1:],
        mode='constant',
        preserve_range=True
    )
    return resized[np.newaxis, ...]

# Define o diretório raiz dos dados
data_dir = '../../dataset_goiasmuticlasse4claymodel/data/raw'

# Lista para armazenar os caminhos dos arquivos .npz
npz_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.npz')]

# Define as proporções para o split
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Divide os dados em treino, validação e teste
train_files, test_val_files = train_test_split(npz_files, train_size=train_ratio, random_state=42)
val_files, test_files = train_test_split(test_val_files, train_size=val_ratio / (val_ratio + test_ratio), random_state=42)

# Função para processar e salvar imagens e rótulos como .npy
def process_and_split_npz(filepath, output_dir, split):
    """
    Processa um arquivo .npz e salva as imagens e máscaras no formato .npy nos diretórios img e gt.
    Se necessário, ajusta o formato para o esperado.

    Args:
        filepath (str): Caminho do arquivo .npz.
        output_dir (str): Diretório base para salvar os arquivos processados.
        split (str): Nome do conjunto ('train', 'val' ou 'test').
    """
    filename = os.path.basename(filepath).split('.')[0]

    # Carrega os dados do arquivo .npz
    data = np.load(filepath)
    image_array = data['array']

    # Separar a última banda como rótulo e as demais como imagem
    class_array = image_array[-1:, :, :]  # Rótulo (1, H, W)
    image_array = image_array[:-1, :, :]  # Imagem (C-1, H, W)

    # Ajustar o formato da imagem, se necessário
    if image_array.shape != (10, 256, 256):
        image_array = resize_image_array(image_array, (10, 256, 256))

    # Ajustar o formato do rótulo, se necessário
    if class_array.shape != (1, 256, 256):
        class_array = resize_label_array(class_array, (1, 256, 256))

    # Criar diretórios para imagens e rótulos
    img_dir = os.path.join(output_dir, split, "img")
    gt_dir = os.path.join(output_dir, split, "gt")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Salvar como .npy
    np.save(os.path.join(img_dir, f"{filename}.npy"), image_array)
    np.save(os.path.join(gt_dir, f"{filename}.npy"), class_array)


# Define o diretório de saída
output_dir = '../../dataset_goiasmuticlasse4claymodel/data'

# Processa e salva os arquivos de treino
print("Processando conjunto de treino...")
for file in train_files:
    process_and_split_npz(file, output_dir, 'train')

# Processa e salva os arquivos de validação
print("Processando conjunto de validação...")
for file in val_files:
    process_and_split_npz(file, output_dir, 'val')

# Processa e salva os arquivos de teste
print("Processando conjunto de teste...")
for file in test_files:
    process_and_split_npz(file, output_dir, 'test')

print("Preparação do dataset concluída!")
