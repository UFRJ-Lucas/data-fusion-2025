import pandas as pd
import matplotlib.pyplot as plt
import pygame

def carregar_dados_trajetoria(path_gaze="df_gaze_original.pkl", path_mouse="df_mouse_original.pkl"):
    """
    Carrega os DataFrames de gaze e mouse a partir dos arquivos .pkl.
    Retorna: df_gaze, df_mouse
    """
    df_gaze = pd.read_pickle(path_gaze)
    df_mouse = pd.read_pickle(path_mouse)
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

def alinhar_gaze_para_mouse(df_gaze, screen_width, screen_height):
    x_min, x_max = df_gaze['x'].min(), df_gaze['x'].max()
    y_min, y_max = df_gaze['y'].min(), df_gaze['y'].max()
    
    df_gaze_alinhado = df_gaze.copy()
    # Normaliza X e Y
    df_gaze_alinhado['x'] = (df_gaze['x'] - x_min) / (x_max - x_min) * screen_width
    df_gaze_alinhado['y'] = (df_gaze['y'] - y_min) / (y_max - y_min) * screen_height
    
    # Inverte Y
    #df_gaze_alinhado['y'] = screen_height - df_gaze_alinhado['y']

    # Inverte X para coincidir com o sentido do mouse
    df_gaze_alinhado['x'] = screen_width - df_gaze_alinhado['x']
    
    return df_gaze_alinhado

def alinhar_gaze_para_mouse2(df_gaze, screen_width, screen_height, x_offset=400, y_offset=250, x_scale=0.5, y_scale=0.5):
    x_min, x_max = df_gaze['x'].min(), df_gaze['x'].max()
    y_min, y_max = df_gaze['y'].min(), df_gaze['y'].max()
    
    df_gaze_alinhado = df_gaze.copy()
    # Normaliza
    df_gaze_alinhado['x'] = ((df_gaze['x'] - x_min) / (x_max - x_min)) * screen_width
    df_gaze_alinhado['y'] = ((df_gaze['y'] - y_min) / (y_max - y_min)) * screen_height
    
    # Ajusta escala e deslocamento
    df_gaze_alinhado['x'] = (df_gaze_alinhado['x'] * x_scale) + x_offset
    df_gaze_alinhado['y'] = (df_gaze_alinhado['y'] * y_scale) + y_offset
    
    # Inverte eixo X se necessário
    df_gaze_alinhado['x'] = screen_width - df_gaze_alinhado['x']
    
    return df_gaze_alinhado


def mostrar_qtd_registros(df_mouse, df_gaze):
    print(f"Quantidade de registros do mouse: {len(df_mouse)}")
    print(f"Quantidade de registros do gaze: {len(df_gaze)}")

df_gaze, df_mouse = carregar_dados_trajetoria()

mostrar_qtd_registros(df_mouse, df_gaze)

# Apenas para pegar o mesmo tamanho de tela do momento da captura
pygame.init() 
screen_info = pygame.display.Info()
screen_width = int(0.8*screen_info.current_w)
screen_height = int(0.8*screen_info.current_h)
pygame.quit()
##########

df_gaze_alinhado = alinhar_gaze_para_mouse2(df_gaze, screen_width, screen_height)

plot_trajetoria(df_gaze_alinhado, df_mouse)