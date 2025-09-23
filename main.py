import pandas as pd
import matplotlib.pyplot as plt
import pygame

def mostrar_qtd_registros(df_mouse, df_gaze):
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

def plot_trajetoria(df_gaze, df_mouse, figsize=(12, 8), gaze_color='red', mouse_color='blue'):
    """
    Plota trajetórias do gaze e do mouse em um único gráfico.
    """
    plt.figure(figsize=figsize)
    
    if not df_gaze.empty:
        plt.plot(df_gaze['x'], df_gaze['y'], color=gaze_color, marker='o', markersize=3, label='Gaze')
    if not df_mouse.empty:
        plt.plot(df_mouse['x'], df_mouse['y'], color=mouse_color, marker='x', markersize=3, label='Mouse')
    
    plt.gca().invert_yaxis()  # Coordenadas de tela: (0,0) topo esquerdo
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajetória Gaze vs Mouse')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df_gaze, df_mouse = carregar_dados_trajetoria()
    plot_trajetoria(df_gaze, df_mouse)