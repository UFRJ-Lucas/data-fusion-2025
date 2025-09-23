import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

def aplicar_kalman_filter(df_mouse, df_gaze):
    """
    Aplica o filtro de Kalman nas coordenadas x e y do DataFrame do mouse.
    df_mouse: DataFrame de movimentos do mouse
    df_gaze: DataFrame de movimentos do gaze (usado para ajustar o Kalman Filter)
    """
    # TODO: Aplicar Kalman Filter
    df_mouse['x_filtered'] = df_mouse['x_noisy']
    df_mouse['y_filtered'] = df_mouse['y_noisy']

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
        plt.plot(df_mouse['x_noisy'], df_mouse['y_noisy'], color='orange', marker='.', markersize=2, label='Mouse with Noise')
        plt.plot(df_mouse['x_filtered'], df_mouse['y_filtered'], color='green', marker='.', markersize=2, label='Mouse Filtered')
    
    plt.gca().invert_yaxis()  # Coordenadas de tela: (0,0) topo esquerdo
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaze vs Mouse trajectory')
    plt.legend()
    plt.show()

def calcular_erros(df_mouse):
    """
    Calcula os erros entre os dados do mouse (ground truth) e os dados com ruído e filtrados.
    Retorna os erros médios absolutos (MAE) e os erros quadráticos médios (RMSE).
    """
    if 'x_noisy' not in df_mouse or 'x_filtered' not in df_mouse:
        print("Dados insuficientes para calcular erros.")
        return

    # Erros para dados com ruído
    mae_noisy = mean_absolute_error(df_mouse['x'], df_mouse['x_noisy']) + \
                mean_absolute_error(df_mouse['y'], df_mouse['y_noisy'])
    rmse_noisy = np.sqrt(mean_squared_error(df_mouse['x'], df_mouse['x_noisy']) + \
                         mean_squared_error(df_mouse['y'], df_mouse['y_noisy']))

    # Erros para dados filtrados
    mae_filtered = mean_absolute_error(df_mouse['x'], df_mouse['x_filtered']) + \
                   mean_absolute_error(df_mouse['y'], df_mouse['y_filtered'])
    rmse_filtered = np.sqrt(mean_squared_error(df_mouse['x'], df_mouse['x_filtered']) + \
                            mean_squared_error(df_mouse['y'], df_mouse['y_filtered']))

    print(f"Erro Médio Absoluto (MAE) - Noisy: {mae_noisy:.4f}, Filtered: {mae_filtered:.4f}")
    print(f"Erro Quadrático Médio (RMSE) - Noisy: {rmse_noisy:.4f}, Filtered: {rmse_filtered:.4f}")

    return mae_noisy, mae_filtered, rmse_noisy, rmse_filtered

if __name__ == "__main__":
    df_gaze, df_mouse = carregar_dados_trajetoria() # Carrega os dados
    df_mouse = gerar_ruido_aleatorio(df_mouse, sigma_min=2, sigma_max=30)  # Adiciona ruído ao mouse
    df_mouse = aplicar_kalman_filter(df_mouse, df_gaze)  # Aplica o Kalman Filter
    calcular_erros(df_mouse)  # Calcula e exibe os erros
    plot_trajetoria(df_gaze, df_mouse)  # Plota as trajetórias