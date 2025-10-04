import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Função 1: Ler um arquivo .pkl e retornar o DataFrame ---

def carregar_dataframe(caminho_arquivo):
    """
    Lê um arquivo .pkl e o carrega como um DataFrame do pandas.

    Args:
        caminho_arquivo (str): O caminho para o arquivo .pkl.

    Returns:
        pd.DataFrame or None: O DataFrame carregado ou None se o arquivo não for encontrado.
    """
    # Verifica se o arquivo existe antes de tentar carregar
    if not os.path.exists(caminho_arquivo):
        print(f"Erro: O arquivo '{caminho_arquivo}' não foi encontrado.")
        return None
        
    try:
        df = pd.read_pickle(caminho_arquivo)
        print(f"Arquivo '{caminho_arquivo}' carregado com sucesso. {len(df)} registros.")
        return df
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo '{caminho_arquivo}': {e}")
        return None

# --- Função 2: Mostrar as primeiras 5 linhas de um DataFrame ---

def mostrar_cabecalho(df, nome_df="DataFrame"):
    """
    Exibe as 5 primeiras linhas (cabeçalho) de um DataFrame.

    Args:
        df (pd.DataFrame): O DataFrame a ser exibido.
        nome_df (str): Um nome descritivo para o DataFrame (opcional).
    """
    if df is not None and isinstance(df, pd.DataFrame):
        print(f"\n--- Cabeçalho de '{nome_df}' ---")
        print(df.head())
        print("-" * (len(nome_df) + 20))
    else:
        print(f"Não é possível mostrar o cabeçalho de '{nome_df}' pois o DataFrame é inválido ou vazio.")

# --- Função 3: Plotar as trajetórias para comparação ---

def plotar_trajetorias(dataframes_dict):
    """
    Plota as trajetórias 'x' vs 'y' de múltiplos DataFrames em um único gráfico.

    Args:
        dataframes_dict (dict): Um dicionário onde as chaves são os rótulos (str) 
                                e os valores são os DataFrames (pd.DataFrame).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cores = {'Mouse': 'blue', 'Olhar': 'red', 'Estabilizado': 'cyan'}
    
    for label, df in dataframes_dict.items():
        if df is not None and not df.empty and 'x' in df.columns and 'y' in df.columns:
            ax.plot(df['x'], df['y'], marker='.', linestyle='-', label=label, color=cores.get(label, None))
        else:
            print(f"Aviso: O DataFrame para '{label}' está vazio ou não contém as colunas 'x' e 'y'. Não será plotado.")
            
    ax.set_title('Comparação de Trajetórias: Mouse vs Olhar vs Estabilizado')
    ax.set_xlabel('Coordenada X (pixels)')
    ax.set_ylabel('Coordenada Y (pixels)')
    
    # Inverte o eixo Y para corresponder às coordenadas da tela (onde 0,0 é no canto superior esquerdo)
    ax.invert_yaxis()
    
    ax.legend()
    ax.grid(True)
    plt.axis('equal') # Garante que a escala dos eixos X e Y seja a mesma
    plt.show()


# --- Como Usar as Funções ---

if __name__ == "__main__":
    # 1. Defina os nomes dos arquivos que você salvou
    arquivo_mouse = "df_mouse_original.pkl"
    arquivo_gaze = "df_gaze_original.pkl"
    arquivo_final = "df_cursor_final.pkl" # Verifique se o nome do arquivo está correto

    # 2. Carregue os DataFrames usando a primeira função
    df_mouse = carregar_dataframe(arquivo_mouse)
    df_gaze = carregar_dataframe(arquivo_gaze)
    df_final = carregar_dataframe(arquivo_final)
    
    # 3. Mostre as primeiras 5 linhas de um dos DataFrames para verificar
    mostrar_cabecalho(df_final, nome_df="Cursor Estabilizado")
    
    # 4. Crie o dicionário e plote o gráfico de comparação
    if df_mouse is not None and df_gaze is not None and df_final is not None:
        trajetorias = {
            'Mouse': df_mouse,
            'Olhar': df_gaze,
            'Estabilizado': df_final
        }
        plotar_trajetorias(trajetorias)