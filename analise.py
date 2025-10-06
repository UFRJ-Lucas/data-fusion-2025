import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Funções de Análise (sem alterações em seu conteúdo) ---

def carregar_dataframe(caminho_arquivo):
    """Lê um arquivo .pkl e o carrega como um DataFrame do pandas."""
    if not os.path.exists(caminho_arquivo): print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado."); return None
    try:
        df = pd.read_pickle(caminho_arquivo)
        print(f"Arquivo '{caminho_arquivo}' carregado com sucesso. {len(df)} registros.")
        return df
    except Exception as e: print(f"Ocorreu um erro ao ler o arquivo '{caminho_arquivo}': {e}"); return None

def calcular_erros_de_fidelidade(df_ground_truth, df_comparacao, nome_comparacao):
    """Calcula e imprime o MAE, RMSE e Erro Máximo entre duas trajetórias."""
    if df_ground_truth is None or df_comparacao is None or df_ground_truth.empty or df_comparacao.empty: return
    print(f"\n--- Métricas de Fidelidade para: {nome_comparacao} ---")
    tamanho_minimo = min(len(df_ground_truth), len(df_comparacao))
    if len(df_ground_truth) != len(df_comparacao): print(f"Aviso: Tamanhos diferentes. Comparando os primeiros {tamanho_minimo} pontos.")
    gt, comp = df_ground_truth.iloc[:tamanho_minimo], df_comparacao.iloc[:tamanho_minimo]
    distancias = np.sqrt((gt['x'] - comp['x'])**2 + (gt['y'] - comp['y'])**2)
    mae, rmse, erro_max = np.mean(distancias), np.sqrt(np.mean(distancias**2)), np.max(distancias)
    print(f"  - Erro Médio Absoluto (MAE):    {mae:.2f} pixels")
    print(f"  - Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f} pixels")
    print(f"  - Erro Máximo:                {erro_max:.2f} pixels")

def analisar_qualidade_caminho(df, nome_trajetoria):
    """Calcula e imprime o Comprimento Total e a Sacudida Total de uma trajetória."""
    if df is None or df.empty or len(df) < 2: print(f"\nNão foi possível analisar a qualidade de '{nome_trajetoria}'."); return
    print(f"\n--- Métricas de Qualidade de Caminho para: {nome_trajetoria} ---")
    pontos = df[['x', 'y']].to_numpy()
    segmentos = np.linalg.norm(np.diff(pontos, axis=0), axis=1)
    comprimento_total = np.sum(segmentos)
    print(f"  - Comprimento Total do Caminho: {comprimento_total:.2f} pixels")
    if len(df) > 2:
        vetores = np.diff(pontos, axis=0); angulos = np.arctan2(vetores[:, 1], vetores[:, 0])
        mudancas_de_angulo = np.abs(np.diff(angulos))
        mudancas_de_angulo = np.minimum(mudancas_de_angulo, 2 * np.pi - mudancas_de_angulo)
        sacudida_total = np.sum(mudancas_de_angulo)
        print(f"  - 'Sacudida' Total (Soma de Ângulos): {sacudida_total:.2f} radianos")

def plotar_trajetorias(dataframes_dict, titulo_grafico):
    """Plota as trajetórias de múltiplos DataFrames em um único gráfico."""
    fig, ax = plt.subplots(figsize=(16, 10))
    # MODIFICAÇÃO: Dicionário de cores atualizado para os dois filtros
    cores = {
        'Mouse Real': 'green', 
        'Mouse com Tremor': 'blue', 
        'Olhar': 'red', 
        'Estabilizado (Freio)': 'orange', 
        'Estabilizado (Kalman)': 'cyan'
    }
    for label, df in dataframes_dict.items():
        if df is not None and not df.empty and 'x' in df.columns and 'y' in df.columns:
            ax.plot(df['x'], df['y'], marker='.', markersize=4, linestyle='-', label=label, color=cores.get(label, 'black'), linewidth=1.5)
    ax.set_title(titulo_grafico); ax.set_xlabel('Coordenada X (pixels)'); ax.set_ylabel('Coordenada Y (pixels)')
    ax.invert_yaxis(); ax.legend(); ax.grid(True); plt.axis('equal'); plt.show()


# --- Bloco Principal (Simplificado e Unificado) ---

if __name__ == "__main__":

    OUTPUT_DIR = "./resultados/"
    
    # MODIFICAÇÃO: Define o conjunto único de arquivos da execução unificada
    arquivos = {
        "Mouse Real": OUTPUT_DIR + "df_mouse_real_unificado.pkl",
        "Mouse com Tremor": OUTPUT_DIR + "df_mouse_com_tremor_unificado.pkl",
        "Olhar": OUTPUT_DIR + "df_gaze_original_unificado.pkl",
        "Estabilizado (Freio)": OUTPUT_DIR + "df_final_freio_adaptativo.pkl",
        "Estabilizado (Kalman)": OUTPUT_DIR + "df_final_kalman.pkl"
    }

    titulo_analise = "Comparativo Final: Freio Adaptativo vs. Kalman"

    print("#"*60)
    print(f"# ANÁLISE DO CONJUNTO: {titulo_analise.upper()}")
    print("#"*60 + "\n")

    # Carrega todos os DataFrames
    dataframes = {label: carregar_dataframe(path) for label, path in arquivos.items()}
    
    # Exibe relatório de métricas de fidelidade
    print("\n" + "="*50)
    print("RELATÓRIO DE MÉTRICAS DE FIDELIDADE (Distância ao Mouse Real)")
    print("="*50)
    if dataframes.get("Mouse Real") is not None:
        # Compara cada resultado com o "Mouse Real"
        calcular_erros_de_fidelidade(dataframes.get("Mouse com Tremor"), dataframes["Mouse Real"], "Mouse com Tremor")
        calcular_erros_de_fidelidade(dataframes.get("Estabilizado (Freio)"), dataframes["Mouse Real"], "Estabilizado (Freio)")
        calcular_erros_de_fidelidade(dataframes.get("Estabilizado (Kalman)"), dataframes["Mouse Real"], "Estabilizado (Kalman)")
    
    # Exibe relatório de métricas de qualidade de caminho
    print("\n" + "="*50)
    print("RELATÓRIO DE MÉTRICAS DE QUALIDADE DE CAMINHO")
    print("="*50)
    for label, df in dataframes.items():
        if label != 'Olhar':
            analisar_qualidade_caminho(df, label)
    print("\n" + "="*50)
    
    # Plota o gráfico
    plotar_trajetorias(dataframes, f"Análise de Trajetórias - {titulo_analise}")