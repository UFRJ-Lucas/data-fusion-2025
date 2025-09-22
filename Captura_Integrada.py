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

# Offsets dos pontos
x_offset_inicio, y_offset_inicio = -150, 150
x_offset_intermediario, y_offset_intermediario = 0, -120
x_offset_fim, y_offset_fim = 0, 50

# Variáveis globais
eventos = []
ultimo_click = {}
capturando = False

# Constantes de cores
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
BLANK = (0,0,0)
WHITE = (255, 255, 255)

# -----------------------------
# Registrar evento do mouse
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
            registrar_evento(x, y, 'double_click', button, 2)
            return 'double_click', 2
    # Clique simples
    ultimo_click[key] = {'timestamp': now, 'x': x, 'y': y}
    registrar_evento(x, y, 'click', button, 1)
    return 'click', 1


# -----------------------------
# Inicialização
# -----------------------------
def init_gaze_screen(scale=0.8, font_size=48):
    pygame.init()
    pygame.font.init()
    
    screen_info = pygame.display.Info()
    screen_width = int(scale*screen_info.current_w)
    screen_height = int(scale*screen_info.current_h)
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Integração Mouse + Câmera")
    
    bold_font = pygame.font.Font(None, font_size)
    bold_font.set_bold(True)
    
    gestures = EyeGestures_v3()
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
# Loop principal de captura
# -----------------------------
def gaze_main_loop(gestures, video_capture, screen, screen_width, screen_height, bold_font, n_points=25, context="my_context"):
    clock = pygame.time.Clock()
    running = True
    iterator = 0
    prev_x, prev_y = 0, 0
    capturing_input = False

    # Pontos de clique
    ponto_inicio = (screen_width // 4 + x_offset_inicio, screen_height // 2 + y_offset_inicio)
    ponto_intermediario = (screen_width // 2 + x_offset_intermediario, screen_height // 2 + y_offset_intermediario)
    ponto_fim = (3 * screen_width // 4 + x_offset_fim, screen_height // 2 + y_offset_fim)

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

                if not capturing_input and click_type == 'double_click' and hypot(x - ponto_inicio[0], y - ponto_inicio[1]) <= POINT_RADIUS:
                    capturing_input = True
                    print("Captura iniciada")
                elif capturing_input and click_type == 'double_click' and hypot(x - ponto_fim[0], y - ponto_fim[1]) <= POINT_RADIUS:
                    capturing_input = False
                    running = False
                    print("Captura encerrada")
            elif event.type == pygame.MOUSEMOTION and capturing_input:
                x, y = event.pos
                eventos_mouse.append({
                    'timestamp': time.time(),
                    'x': x,
                    'y': y,
                    'event_type': 'move',
                    'button': 'mouse',
                    'click_count': 0
                })

            pygame.draw.circle(screen, GREEN, ponto_inicio, POINT_RADIUS)
            pygame.draw.circle(screen, BLUE, ponto_intermediario, POINT_RADIUS)
            pygame.draw.circle(screen, RED, ponto_fim, POINT_RADIUS)

            if capturing_input and gaze_event is not None:
                eventos_gaze.append({
                    'timestamp': time.time(),
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
    gestures, video_capture, screen, w, h, bold_font = init_gaze_screen()
    n_points = setup_calibration(gestures, max_points=25)
    gaze_main_loop(gestures, video_capture, screen, w, h, bold_font, n_points=n_points)
    finalize_gaze(video_capture)