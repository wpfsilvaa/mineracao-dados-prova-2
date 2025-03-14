import numpy as np

def gerar_patterns():
    patterns = np.array([
        [0,0,1,1,1,0,0,
         0,1,0,0,0,1,0,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         0,1,0,0,0,1,0,
         0,0,1,1,1,0,0],
        [0,0,0,1,0,0,0,
         0,0,1,1,0,0,0,
         0,1,0,1,0,0,0,
         0,0,0,1,0,0,0,
         0,0,0,1,0,0,0,
         0,0,0,1,0,0,0,
         0,0,0,1,0,0,0,
         0,0,0,1,0,0,0,
         0,1,1,1,1,1,0],
        [0,1,1,1,1,1,0,
         1,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         0,0,0,0,0,1,0,
         0,0,0,0,1,0,0,
         0,0,0,1,0,0,0,
         0,0,1,0,0,0,0,
         0,1,0,0,0,0,0,
         0,1,1,1,1,1,1],
        [0,1,1,1,1,1,0,
         1,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         0,0,1,1,1,1,0,
         0,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         0,1,1,1,1,1,0],
        [0,0,0,1,0,0,1,
         0,0,1,0,0,0,1,
         0,0,1,0,0,0,1,
         0,1,0,0,0,0,1,
         0,1,1,1,1,1,0,
         0,0,0,0,0,1,0,
         0,0,0,0,0,1,0,
         0,0,0,0,0,1,0,
         0,0,0,0,1,1,1],
        [0,1,1,1,1,1,1,
         0,1,0,0,0,0,0,
         0,1,0,0,0,0,0,
         0,1,0,0,0,0,0,
         0,1,1,1,1,1,0,
         0,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         0,1,1,1,1,1,0],
        [0,0,1,1,1,1,0,
         0,1,0,0,0,0,1,
         1,0,0,0,0,0,0,
         1,0,0,0,0,0,0,
         1,1,1,1,1,1,0,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         1,1,0,0,0,1,1,
         0,1,1,1,1,1,0],
        [1,1,1,1,1,1,0,
         0,0,0,0,0,1,0,
         0,0,0,0,0,1,0,
         0,0,0,0,1,0,0,
         0,0,0,0,1,0,0,
         0,0,0,1,0,0,0,
         0,0,0,1,0,0,0,
         0,0,1,0,0,0,0,
         0,0,1,0,0,0,0],
        [0,1,1,1,1,1,0,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         0,1,1,1,1,1,0,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         0,1,1,1,1,1,0],
         [0,1,1,1,1,1,0,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         1,0,0,0,0,0,1,
         0,1,1,1,1,1,1,
         0,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         0,0,0,0,0,0,1,
         0,1,1,1,1,1,0],

    ])
    return patterns

def adicionar_ruido(patterns, noise_level):
    ruido_pattern = np.copy(patterns)
    num_elements = patterns.shape[1]
    for i in range(patterns.shape[0]):
        indices = np.random.choice(num_elements, int(noise_level * num_elements), replace=False)
        ruido_pattern[i, indices] = 1 - ruido_pattern[i, indices]
    return ruido_pattern

if __name__ == '__main__':
    gerar_patterns()
