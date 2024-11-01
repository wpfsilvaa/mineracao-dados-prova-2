import tensorflow as tf
import numpy as np
from gerador import gerar_patterns,adicionar_ruido

def carregar_modelo(nome):
    modelo_carregado = tf.keras.models.load_model(nome)
    return modelo_carregado

def testar_modelo(model, patterns):
    predictions = model.predict(patterns)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

def calcular_precisao(predictions, expected):
    acertos = np.sum(predictions == expected)
    total = len(expected)
    precisao = acertos / total * 100
    erro = 100 - precisao
    return precisao, erro

labels_esperados = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arquivos = ["modelo_numeros_RNA0.keras","modelo_numeros_RNA1.keras","modelo_numeros_RNA2.keras","modelo_numeros_RNA3.keras"]
for arquivo in arquivos:
    print(f"Testando: {arquivo}")
    model = carregar_modelo(arquivo)

    patterns = gerar_patterns()
    niveis_de_ruido = [0.0, 0.1, 0.2, 0.3]
    lista_patterns_ruidos = []

    print(f"\nTestando o modelo ({arquivo}) com os padrões originais:")
    original_predictions = testar_modelo(model, patterns)
    
    precisao, erro = calcular_precisao(original_predictions, labels_esperados)
    print("Predições para padrões originais:", original_predictions)
    print(f"Precisão: {precisao:.2f}%, Erro: {erro:.2f}%")

    for i, nivel_de_ruido in enumerate(niveis_de_ruido):
        noisy_patterns = adicionar_ruido(patterns, nivel_de_ruido)
        lista_patterns_ruidos.append(noisy_patterns)

        print(f"\nTestando o modelo ({arquivo}) com ruído de {nivel_de_ruido * 100:.0f}%:")
        noisy_predictions = testar_modelo(model, noisy_patterns)

        precisao, erro = calcular_precisao(noisy_predictions, labels_esperados)
        print(f"Predições para padrões com ruído de {nivel_de_ruido * 100:.0f}%:", noisy_predictions)
        print(f"Precisão: {precisao:.2f}%, Erro: {erro:.2f}%")

        # for expectativa, realidade in zip(labels_esperados, noisy_predictions):
        #     print(f"Esperado: {expectativa}, Exibido: {realidade}")

