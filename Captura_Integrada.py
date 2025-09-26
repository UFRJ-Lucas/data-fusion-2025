# Geral
import pygame, time
import numpy as np
import pandas as pd
from math import hypot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3


# -----------------------------
# Parâmetros
# -----------------------------
DOUBLE_CLICK_MAX_INTERVAL = 0.3
POINT_RADIUS = 20
GAZE_POINT_RADIUS = 30

# Variáveis globais
eventos = []
ultimo_click = {}
capturando = False

# Constantes de cores
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
BLANK = (125,125,125)
WHITE = (255, 255, 255)

# -----------------------------
# Registrar evento do mouse
# -----------------------------
def register_event(x, y, event_type, button, click_count):
    eventos.append({
        'timestamp': time.time(),
        'x': x,
        'y': y,
        'event_type': event_type,
        'button': button,
        'click_count': click_count
    })

# -----------------------------
# Detectar clique
# -----------------------------
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
            register_event(x, y, 'double_click', button, 2)
            return 'double_click', 2
    # Clique simples
    ultimo_click[key] = {'timestamp': now, 'x': x, 'y': y}
    register_event(x, y, 'click', button, 1)
    return 'click', 1


# -----------------------------
# Inicialização
# -----------------------------
def init_gaze_screen(scale=0.8, font_size=48, calibration_radius=300):
    pygame.init()
    pygame.font.init()
    
    screen_info = pygame.display.Info()
    screen_width = int(scale*screen_info.current_w)
    screen_height = int(scale*screen_info.current_h)
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Integração Mouse + Câmera")
    
    bold_font = pygame.font.Font(None, font_size)
    bold_font.set_bold(True)
    
    gestures = EyeGestures_v3(calibration_radius=calibration_radius)
    video_capture = VideoCapture(0)
    
    return gestures, video_capture, screen, screen_width, screen_height, bold_font

# -----------------------------
# Configuração da calibração
# -----------------------------
def setup_calibration(gestures, max_points=25, context="my_context"):
    x = np.arange(0, 1.1, 0.2)
    y = np.arange(0, 1.1, 0.2)
    xx, yy = np.meshgrid(x, y)
    
    calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
    n_points = min(len(calibration_map), max_points)
    np.random.shuffle(calibration_map)
    
    gestures.uploadCalibrationMap(calibration_map, context=context)
    gestures.setFixation(1.0)
    
    return n_points

# -----------------------------
# Criação de pontos de clique
# -----------------------------
def make_point(x_percent, y_percent, screen_width, screen_height):
    """
    Cria um ponto na tela baseado em porcentagens da largura e altura da tela.
        x_percent: float entre 0 e 1 representando a posição horizontal
        y_percent: float entre 0 e 1 representando a posição vertical
    """
    x_pos = x_percent * screen_width
    y_pos = y_percent * screen_height

    return (x_pos, y_pos)


# -----------------------------
# Loop principal de captura
# -----------------------------
def gaze_main_loop(gestures, video_capture, screen, screen_width, screen_height, bold_font, n_points=25, context="my_context"):
    clock = pygame.time.Clock()
    running = True
    iterator = 0
    prev_x, prev_y = 0, 0
    capturing_input = False

    # Pontos de clique
    point_positions = [(0.3, 0.5),(0.5, 0.25),(0.75, 0.55),(0,1),(0,0),(1,1),(1,0),(0.5,0.5)] # primeiro ponto é inicio, ultimo é fim
    click_points = []
    for pos in point_positions:
        click_points.append(make_point(pos[0], pos[1], screen_width, screen_height))
    if len(click_points) < 2:
        pygame.quit()
        raise ValueError("Defina pelo menos dois pontos de clique.")
    
    # Pontos de inicio e fim
    start_point = click_points[0]
    end_point = click_points[-1]

    # DataFrames
    eventos_mouse = []
    eventos_gaze = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or (event.key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL):
                    running = False
                    return
        
        if not running:
            break
        
        ret, frame = video_capture.read()
        if not ret: # TODO: Verificar se funciona
            frame.release()
            running = False  # Encerrar loop imediatamente se falha na captura
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1)
        
        calibrate = (iterator <= n_points)
        gaze_event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context=context)
        
        if gaze_event is None and calibration is None:
            continue
        
        screen.fill(BLANK)
        
        # Mostrar câmera reduzida no canto
        screen.blit(
            pygame.surfarray.make_surface(np.rot90(gaze_event.sub_frame)),
            (0, 0)
        )
        my_font = pygame.font.SysFont('Comic Sans MS', 30)
        text_surface = my_font.render(f'{gaze_event.fixation}', False, (0, 0, 0))
        screen.blit(text_surface, (0,0))
        
        # Visualização da calibração
        if calibrate and calibration is not None:
            if calibration.point[0] != prev_x or calibration.point[1] != prev_y:
                iterator += 1
                prev_x, prev_y = calibration.point[0], calibration.point[1]
            pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)
            text_surface = bold_font.render(f"{iterator}/{n_points}", True, WHITE)
            text_square = text_surface.get_rect(center=calibration.point)
            screen.blit(text_surface, text_square)
        
        # Após calibração
        elif not calibrate:
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos
                button = event.button
                click_type, click_count = detect_click(x, y, button)

                if not capturing_input and click_type == 'double_click' and hypot(x - start_point[0], y - start_point[1]) <= POINT_RADIUS:
                    capturing_input = True
                    print("Captura iniciada")
                elif capturing_input and click_type == 'double_click' and hypot(x - end_point[0], y - end_point[1]) <= POINT_RADIUS:
                    capturing_input = False
                    running = False
                    print("Captura encerrada")
            
            pygame.draw.circle(screen, GREEN, start_point, POINT_RADIUS) # Desenha ponto de início
            pygame.draw.circle(screen, RED, end_point, POINT_RADIUS) # Desenha ponto de fim
            for point in click_points[1:-1]:
                pygame.draw.circle(screen, BLUE, point, POINT_RADIUS) # Desenha pontos intermediários

            # Captura dados do mouse e visão
            if capturing_input and gaze_event is not None:
                timestamp = pygame.time.get_ticks()
                mouse_x, mouse_y = pygame.mouse.get_pos()
                eventos_mouse.append({
                    'timestamp': timestamp,
                    'x': mouse_x,
                    'y': mouse_y,
                    'event_type': 'move',
                    'button': 'mouse',
                    'click_count': 0
                })
                eventos_gaze.append({
                    'timestamp': timestamp,
                    'x': gaze_event.point[0],
                    'y': gaze_event.point[1],
                    'fixation': gaze_event.fixation,
                    'saccades': gaze_event.saccades
                })

        surface = pygame.display.get_surface()
        if surface and gaze_event is not None:
            pygame.draw.circle(surface, RED, gaze_event.point, GAZE_POINT_RADIUS)
            
        pygame.display.flip()

        clock.tick(60)

    df_mouse = pd.DataFrame(eventos_mouse)
    df_gaze = pd.DataFrame(eventos_gaze)
    print(f'Registros capturados do mouse: {len(df_mouse)}')
    print(f'Registros capturados do gaze: {len(df_gaze)}')

    df_mouse.to_pickle("df_mouse_original.pkl")
    df_gaze.to_pickle("df_gaze_original.pkl")

# -----------------------------
# Finalização
# -----------------------------
def finalize_gaze(video_capture):
    pygame.quit()
    video_capture.cap.release()
    del video_capture

# -----------------------------
# Teste do módulo
# -----------------------------
if __name__ == "__main__":
    gestures, video_capture, screen, w, h, bold_font = init_gaze_screen(scale=1) # TODO: ajustar raio de calibração
    n_points = setup_calibration(gestures, max_points=25)
    gaze_main_loop(gestures, video_capture, screen, w, h, bold_font, n_points=n_points)
    finalize_gaze(video_capture)