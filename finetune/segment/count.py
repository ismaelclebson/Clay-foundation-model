import os
import numpy as np
from collections import Counter

# Diretório principal contendo os subdiretórios train, val e test
base_dir = '../../dataset_goiasmuticlasse4claymodel/data'

# Subdiretórios de interesse
splits = ['test']#, 'val', 'test']

# Função para contar os rótulos em um diretório
def count_labels_in_directory(gt_dir):
    label_counts = Counter()
    
    # Percorre todas as imagens no diretório
    for filename in os.listdir(gt_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(gt_dir, filename)
            label_array = np.load(filepath)  # Carrega o array .npy
            
            # Conta os valores únicos no array de rótulos
            unique, counts = np.unique(label_array, return_counts=True)
            label_counts.update(dict(zip(unique, counts)))
    
    return label_counts

# Contador geral para todos os splits
overall_counts = Counter()

# Processa cada split
for split in splits:
    gt_dir = os.path.join(base_dir, split, 'gt')  # Diretório de ground truth
    if not os.path.exists(gt_dir):
        print(f"Diretório {gt_dir} não encontrado. Ignorando.")
        continue
    
    print(f"Contabilizando rótulos no diretório {gt_dir}...")
    split_counts = count_labels_in_directory(gt_dir)
    
    # Exibe os resultados do split atual
    print(f"Rótulos no {split}: {dict(split_counts)}")
    
    # Atualiza os contadores gerais
    overall_counts.update(split_counts)

# Exibe o resultado geral
print("\nContagem total de rótulos em todas as divisões:")
for label, count in sorted(overall_counts.items()):
    print(f"Rótulo {label}: {count}")
