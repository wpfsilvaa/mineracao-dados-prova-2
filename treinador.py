import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from gerador import gerar_patterns,adicionar_ruido

def treinar():
    patterns = gerar_patterns()
    labels = np.eye(10)[:patterns.shape[0]]
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    models = []

    for i in range(4):
        model = Sequential([
            Dense(128, input_shape=(63,), activation='relu'),
            Dense(64, activation='relu'), # Rectified Linear Unit f(x)=max(0,x)
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        models.append(model)

    for i, model in enumerate(models):
        treino_patterns = [patterns]
        for j in range(1, i + 1):
            ruido_patterns = adicionar_ruido(patterns, noise_levels[j])
            treino_patterns.append(ruido_patterns)

        treino_patterns = np.vstack(treino_patterns)
        treino_labels = np.tile(labels, (len(treino_patterns) // len(labels), 1))
        print(f"Treinando modelo_numeros_RNA{i}.keras")
        model.fit(treino_patterns, treino_labels, epochs=15, batch_size=1, verbose=1)
        model.evaluate(treino_patterns, treino_labels, verbose=1)

    for i, model in enumerate(models):
        model.save(f"modelo_numeros_RNA{i}.keras")
        print(f'Modelo RNA{i} salvo como modelo_numeros_RNA{i}.keras')

if __name__ == '__main__':
    treinar()