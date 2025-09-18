# mouse_capture.py
import pygame
import time
import pandas as pd
from math import hypot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------
# Parâmetros
# -----------------------------
DOUBLE_CLICK_MAX_INTERVAL = 0.3
POINT_RADIUS = 20

# Offsets dos pontos
x_offset_inicio, y_offset_inicio = -150, 150
x_offset_intermediario, y_offset_intermediario = 0, -120
x_offset_fim, y_offset_fim = 0, 50

# Variáveis globais
eventos = []
ultimo_click = {}
capturando = False

# -----------------------------
# Funções
# -----------------------------
def registrar_evento(x, y, event_type, button, click_count):
    eventos.append({
        'timestamp': time.time(),
        'x': x,
        'y': y,
        'event_type': event_type,
        'button': button,
        'click_count': click_count
    })

def detect_click(x, y, button):
    global ultimo_click
    now = time.time()
    key = str(button)
    if key in ultimo_click:
        dt = now - ultimo_click[key]['timestamp']
        dist = hypot(x - ultimo_click[key]['x'], y - ultimo_click[key]['y'])
        if dt <= DOUBLE_CLICK_MAX_INTERVAL and dist <= POINT_RADIUS:
            # Duplo clique
            ultimo_click[key] = {'timestamp': now, 'x': x, 'y': y}
            registrar_evento(x, y, 'double_click', button, 2)
            return 'double_click', 2
    # Clique simples
    ultimo_click[key] = {'timestamp': now, 'x': x, 'y': y}
    registrar_evento(x, y, 'click', button, 1)
    return 'click', 1

def capturar_trajetoria():
    global capturando
    pygame.init()
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w // 2, info.current_h // 2
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulação de Trajetória do Mouse")

    # Pontos
    ponto_inicio = (WIDTH // 4 + x_offset_inicio, HEIGHT // 2 + y_offset_inicio)
    ponto_intermediario = (WIDTH // 2 + x_offset_intermediario, HEIGHT // 2 + y_offset_intermediario)
    ponto_fim = (3 * WIDTH // 4 + x_offset_fim, HEIGHT // 2 + y_offset_fim)

    rodando = True
    while rodando:
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (0, 255, 0), ponto_inicio, POINT_RADIUS)
        pygame.draw.circle(screen, (0, 0, 255), ponto_intermediario, POINT_RADIUS)
        pygame.draw.circle(screen, (255, 0, 0), ponto_fim, POINT_RADIUS)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rodando = False
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos
                button = event.button
                click_type, click_count = detect_click(x, y, button)
                if not capturando and click_type == 'double_click' and hypot(x - ponto_inicio[0], y - ponto_inicio[1]) <= POINT_RADIUS:
                    capturando = True
                    print("Captura iniciada")
                if capturando and click_type == 'double_click' and hypot(x - ponto_fim[0], y - ponto_fim[1]) <= POINT_RADIUS:
                    capturando = False
                    rodando = False
                    print("Captura encerrada")
            elif event.type == pygame.MOUSEMOTION and capturando:
                x, y = event.pos
                registrar_evento(x, y, 'move', 'mouse', 0)

    pygame.quit()
    df = pd.DataFrame(eventos)
    return df, WIDTH, HEIGHT

def gerar_ruido_aleatorio(df_mov, sigma_min=2, sigma_max=15):
    """
    Adiciona ruído aleatório nas coordenadas x e y, com intensidade variável
    de forma aleatória ao longo da trajetória.

    df_mov: DataFrame de movimentos
    sigma_min: desvio padrão mínimo
    sigma_max: desvio padrão máximo
    """
    n = len(df_mov)
    if n == 0:
        return df_mov

    # Intensidade do ruído sorteada aleatoriamente para cada ponto
    np.random.seed(42)  # reprodutibilidade
    sigma_x = np.random.uniform(sigma_min, sigma_max, n)
    sigma_y = np.random.uniform(sigma_min, sigma_max, n)

    df_mov['x_noisy'] = df_mov['x'] + np.random.normal(0, sigma_x)
    df_mov['y_noisy'] = df_mov['y'] + np.random.normal(0, sigma_y)
    return df_mov

def plot_trajetorias(df, df_noisy, WIDTH, HEIGHT):
    fig, ax = plt.subplots(figsize=(WIDTH/100, HEIGHT/100), dpi=100)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.invert_yaxis()

    # Trajetórias
    ax.plot(df[df['event_type'] == 'move']['x'], df[df['event_type'] == 'move']['y'], color='black', label='Original')
    ax.plot(df_noisy[df_noisy['event_type'] == 'move']['x_noisy'], df_noisy[df_noisy['event_type'] == 'move']['y_noisy'], color='orange', label='Com Tremor')

    # Cliques
    for df_plot in [df, df_noisy]:
        for _, row in df_plot[df_plot['event_type'].isin(['click', 'double_click'])].iterrows():
            color = 'blue' if row['event_type'] == 'click' else 'red'
            x_plot = row.get('x_noisy', row['x'])
            y_plot = row.get('y_noisy', row['y'])
            ax.scatter(x_plot, y_plot, color=color, s=50)

    # Legendas com círculos
    clique_patch = Line2D([0], [0], marker='o', color='w', label='Clique Único', markerfacecolor='blue', markersize=8)
    double_patch = Line2D([0], [0], marker='o', color='w', label='Duplo Clique', markerfacecolor='red', markersize=8)
    ax.legend(handles=[ax.get_lines()[0], ax.get_lines()[1], clique_patch, double_patch])
    plt.title("Trajetórias: Original x Com Tremor")
    plt.show()

# -----------------------------
# Execução principal
# -----------------------------
if __name__ == "__main__":
    df, WIDTH, HEIGHT = capturar_trajetoria()
    # Filtrar apenas movimentos
    df_mov = df[df['event_type'] == 'move'].copy()

    # Adicionar ruído variável
    df_mov = gerar_ruido_aleatorio(df_mov, sigma_min=2, sigma_max=30)
    df_cliques = df[df['event_type'].isin(['click', 'double_click'])].copy()
    df_cliques['x_noisy'] = df_cliques['x']
    df_cliques['y_noisy'] = df_cliques['y']

    # Novo DataFrame com movimentos ruidosos + cliques originais
    df_noisy = pd.concat([df_mov, df_cliques], ignore_index=True).sort_values('timestamp')

    # Salvar DataFrames
    df.to_pickle("df_original.pkl")
    df_noisy.to_pickle("df_noisy.pkl")

    # Plotar
    plot_trajetorias(df, df_noisy, WIDTH, HEIGHT)