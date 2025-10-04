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
# A variável 'capturando' é controlada localmente na função de loop.
# capturando = False 

# Constantes de cores
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
BLANK = (125,125,125)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)


# -----------------------------
# Suavizar o mouse com média móvel
# -----------------------------
class PositionSmoother:
    """Suaviza uma posição 2D usando média móvel."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history_x = []
        self.history_y = []

    def smooth(self, x, y):
        self.history_x.append(x)
        self.history_y.append(y)
        if len(self.history_x) > self.window_size:
            self.history_x.pop(0)
            self.history_y.pop(0)
        
        avg_x = sum(self.history_x) / len(self.history_x)
        avg_y = sum(self.history_y) / len(self.history_y)
        return np.array([avg_x, avg_y])

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
            ultimo_click[key] = {'timestamp': now, 'x': x, 'y': y}
            register_event(x, y, 'double_click', button, 2)
            return 'double_click', 2
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
    x_pos = x_percent * screen_width
    y_pos = y_percent * screen_height
    return (x_pos, y_pos)

# -----------------------------
# Loop principal de captura (COM LÓGICA DE ATRAÇÃO)
# -----------------------------
def gaze_main_loop(gestures, video_capture, screen, screen_width, screen_height, bold_font, n_points=25, context="my_context"):
    clock = pygame.time.Clock()
    running = True
    iterator = 0
    prev_x, prev_y = 0, 0
    capturing_input = False

    point_positions = [(0.3, 0.5),(0.5, 0.25),(0.75, 0.55),(0.1,0.9),(0.1,0.1),(0.9,1),(0.5,0.5)]
    click_points = [make_point(pos[0], pos[1], screen_width, screen_height) for pos in point_positions]
    start_point, end_point = click_points[0], click_points[-1]

    eventos_mouse, eventos_gaze, eventos_final = [], [], []

    # Inicialização da nova lógica de atração
    mouse_smoother = PositionSmoother(window_size=4)
    final_cursor_pos = np.array(pygame.mouse.get_pos(), dtype=float)
    
    # --- PARÂMETROS DE AJUSTE ---
    # Força com que o olhar "puxa" o mouse. Valores BAIXOS são melhores.
    gaze_pull_strength = 0.15 
    # Suavização do movimento final. Previne "saltos".
    cursor_lerp_factor = 0.6  

    while running:
        event = pygame.event.poll()
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False; return
        
        if not running: break
        
        ret, frame = video_capture.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = np.flip(frame, axis=1)
        
        calibrate = (iterator <= n_points)
        gaze_event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context=context)
        
        if gaze_event is None and calibration is None: continue
        
        screen.fill(BLANK)
        
        if gaze_event and gaze_event.sub_frame is not None:
            screen.blit(pygame.surfarray.make_surface(np.rot90(gaze_event.sub_frame)), (0, 0))
        
        if calibrate and calibration is not None:
            if calibration.point[0] != prev_x or calibration.point[1] != prev_y:
                iterator += 1; prev_x, prev_y = calibration.point[0], calibration.point[1]
            pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)
            text_surface = bold_font.render(f"{iterator}/{n_points}", True, WHITE)
            text_square = text_surface.get_rect(center=calibration.point)
            screen.blit(text_surface, text_square)
        
        elif not calibrate:
            # Lógica de clique e encerramento
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos; button = event.button
                click_type, _ = detect_click(x, y, button)
                if not capturing_input and click_type == 'double_click' and hypot(x-start_point[0], y-start_point[1]) <= POINT_RADIUS:
                    capturing_input = True; print("Captura iniciada")
                elif capturing_input and click_type == 'double_click' and hypot(x-end_point[0], y-end_point[1]) <= POINT_RADIUS:
                    capturing_input = False; running = False; print("Captura encerrada")
            
            # Desenho da UI
            pygame.draw.circle(screen, GREEN, start_point, POINT_RADIUS)
            pygame.draw.circle(screen, RED, end_point, POINT_RADIUS)
            for point in click_points[1:-1]: pygame.draw.circle(screen, BLUE, point, POINT_RADIUS)

            # Lógica de atração que previne o afastamento
            if gaze_event is not None and gaze_event.point is not None:
                mouse_pos = np.array(pygame.mouse.get_pos())
                gaze_pos = np.array(gaze_event.point)
                
                # 1. Âncora: Posição suavizada do mouse. O cursor final não pode se afastar daqui.
                smoothed_mouse_pos = mouse_smoother.smooth(mouse_pos[0], mouse_pos[1])
                
                # 2. Vetor de Atração: A direção que o olhar "puxa".
                attraction_vector = gaze_pos - smoothed_mouse_pos
                
                # 3. Ponto Alvo: O ponto final ideal, que é a âncora mais um pouco na direção do olhar.
                target_pos = smoothed_mouse_pos + gaze_pull_strength * attraction_vector
                
                # 4. Movimento Suave: Interpola linearmente a posição atual do cursor final em direção ao alvo.
                # Isso cria um movimento fluido e amortecido, como se houvesse um elástico.
                final_cursor_pos = (1 - cursor_lerp_factor) * final_cursor_pos + cursor_lerp_factor * target_pos

            # Captura de dados
            if capturing_input and gaze_event is not None:
                timestamp = pygame.time.get_ticks()
                mouse_x, mouse_y = pygame.mouse.get_pos()
                eventos_mouse.append({'timestamp': timestamp, 'x': mouse_x, 'y': mouse_y})
                eventos_gaze.append({'timestamp': timestamp, 'x': gaze_event.point[0], 'y': gaze_event.point[1]})
                eventos_final.append({'timestamp': timestamp, 'x': final_cursor_pos[0], 'y': final_cursor_pos[1]})

        # Desenho dos cursores
        surface = pygame.display.get_surface()
        if surface and gaze_event is not None and gaze_event.point is not None:
            # Olhar bruto (vermelho)
            pygame.draw.circle(surface, RED, gaze_event.point, GAZE_POINT_RADIUS)
            if not calibrate:
                # Cursor final (ciano)
                pygame.draw.circle(surface, CYAN, (int(final_cursor_pos[0]), int(final_cursor_pos[1])), 15)
            
        pygame.display.flip()
        clock.tick(60)

    # Salvar dados
    df_mouse = pd.DataFrame(eventos_mouse)
    df_gaze = pd.DataFrame(eventos_gaze)
    df_final = pd.DataFrame(eventos_final)
    print(f'Registros: Mouse={len(df_mouse)}, Gaze={len(df_gaze)}, Final={len(df_final)}')
    df_mouse.to_pickle("df_mouse_original.pkl")
    df_gaze.to_pickle("df_gaze_original.pkl")
    df_final.to_pickle("df_cursor_final.pkl")

# -----------------------------
# Finalização
# -----------------------------
def finalize_gaze(video_capture):
    pygame.quit()
    if video_capture and video_capture.cap:
        video_capture.cap.release()
    del video_capture

# -----------------------------
# Teste do módulo
# -----------------------------
if __name__ == "__main__":
    gestures, video_capture, screen, w, h, bold_font = init_gaze_screen(scale=1)
    n_points = setup_calibration(gestures, max_points=25)
    gaze_main_loop(gestures, video_capture, screen, w, h, bold_font, n_points=n_points)
    finalize_gaze(video_capture)