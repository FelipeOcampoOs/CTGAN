# -*- coding: utf-8 -*-
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sklearn.model_selection import ParameterGrid
import pandas as pd
data= ('database_non-shows (2) (4).xlsx')
# Definir la metadata del DataFrame
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

param_grid = {
    'epochs': [300,500],
    'generator_dim': [(128, 128),(256, 256), (512, 512), (512, 256, 128)],
    'discriminator_dim': [(128, 128),(256, 256), (512, 512), (512, 256, 128)],
    'pac': [10],
    'discriminator_decay': [1e-4,1e-6, 1e-8],
    'discriminator_lr': [1e-4, 2e-4, 3e-4],
    'embedding_dim': [64, 128, 256],
    'generator_decay': [1e-4, 1e-6, 1e-8],
    'generator_lr': [1e-4, 2e-4, 3e-4],
    'enforce_rounding': [True]
}

# Crear combinaciones de hiperparámetros
grid = ParameterGrid(param_grid)

best_model = None
best_score = float('-inf')

# Evaluar cada combinación de hiperparámetros
for params in grid:
    model = CTGANSynthesizer(
        metadata=metadata,  # Asegurarse de pasar la metadata al inicializar
        generator_dim=params['generator_dim'],
        discriminator_dim=params['discriminator_dim'],
        epochs=params['epochs'],
        pac=params['pac'],
        discriminator_decay=params['discriminator_decay'],
        discriminator_lr=params['discriminator_lr'],
        embedding_dim=params['embedding_dim'],
        generator_decay=params['generator_decay'],
        generator_lr=params['generator_lr'],
        enforce_rounding=params['enforce_rounding']
    )
    
    # Entrenar el modelo
    model.fit(data)
    
    # Generar el doble de muestras sintéticas en comparación con el tamaño original
    synthetic_data = model.sample(len(data) * 2)
    
    # Evaluar la calidad general utilizando las métricas integradas de SDV
    quality_report = evaluate_quality(
        real_data=data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    # Obtener el puntaje de calidad general
    overall_score = quality_report.get_score()

    # Verificar si es el mejor puntaje encontrado
    if overall_score > best_score:
        best_score = overall_score
        best_model = model
        best_synthetic_data = synthetic_data  # Guardar los mejores datos sintéticos

print("Mejor combinación de hiperparámetros:", best_model.get_parameters())
print("Mejor puntaje de calidad:", best_score)
