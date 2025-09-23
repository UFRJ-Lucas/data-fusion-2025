import pandas as pd
import matplotlib.pyplot as plt
import pygame
import numpy as np

def mostrar_qtd_registros(df_mouse, df_gaze):
    """
    Mostra a quantidade de registros nos DataFrames de mouse e gaze.
    """
    print(f"Quantidade de registros do mouse: {len(df_mouse)}")
    print(f"Quantidade de registros do gaze: {len(df_gaze)}")

def carregar_dados_trajetoria(path_gaze="df_gaze_original.pkl", path_mouse="df_mouse_original.pkl"):
    """
    Carrega os DataFrames de gaze e mouse a partir dos arquivos .pkl.
    Retorna: df_gaze, df_mouse
    """
    df_gaze = pd.read_pickle(path_gaze)
    df_mouse = pd.read_pickle(path_mouse)
    mostrar_qtd_registros(df_mouse, df_gaze)
    
    return df_gaze, df_mouse

def gerar_ruido_aleatorio(df_mouse, sigma_min=2, sigma_max=15):
    """
    Adiciona ruído aleatório nas coordenadas x e y, com intensidade variável
    de forma aleatória ao longo da trajetória.

    df_mouse: DataFrame de movimentos
    sigma_min: desvio padrão mínimo
    sigma_max: desvio padrão máximo
    """
    n = len(df_mouse)
    if n == 0:
        return df_mouse

    # Intensidade do ruído sorteada aleatoriamente para cada ponto
    np.random.seed(42)  # reprodutibilidade
    sigma_x = np.random.uniform(sigma_min, sigma_max, n)
    sigma_y = np.random.uniform(sigma_min, sigma_max, n)

    df_mouse['x_noisy'] = df_mouse['x'] + np.random.normal(0, sigma_x)
    df_mouse['y_noisy'] = df_mouse['y'] + np.random.normal(0, sigma_y)
    return df_mouse

def plot_trajetoria(df_gaze, df_mouse, figsize=(12, 8), gaze_color='red', mouse_color='blue'):
    """
    Plota trajetórias do gaze e do mouse em um único gráfico.
    """
    plt.figure(figsize=figsize)
    
    if not df_gaze.empty:
        plt.plot(df_gaze['x'], df_gaze['y'], color=gaze_color, marker='o', markersize=3, label='Gaze')
    if not df_mouse.empty:
        plt.plot(df_mouse['x'], df_mouse['y'], color=mouse_color, marker='x', markersize=3, label='Mouse')
        plt.plot(df_mouse['x_noisy'], df_mouse['y_noisy'], color='orange', marker='.', markersize=2, label='Mouse com Ruído')
    
    plt.gca().invert_yaxis()  # Coordenadas de tela: (0,0) topo esquerdo
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajetória Gaze vs Mouse')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df_gaze, df_mouse = carregar_dados_trajetoria()
    df_mouse = gerar_ruido_aleatorio(df_mouse, sigma_min=2, sigma_max=30)  # Adiciona ruído ao mouse
    plot_trajetoria(df_gaze, df_mouse)