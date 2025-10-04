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

# Constantes de cores
RED = (255, 0, 100)
BLUE = (100, 0, 255)
GREEN = (0, 255, 0)
BLANK = (125,125,125)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)

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
# Detectar tremores com base na velocidade
# -----------------------------

class TremorDetector:
    """Mede a velocidade do mouse para estimar a intensidade do tremor e retorna um peso dinâmico."""
    def __init__(self):
        # Parâmetros de calibração (ajuste estes valores para mudar a sensibilidade)
        self.min_speed_threshold = 2.0   # Abaixo desta velocidade, o tremor é considerado mínimo.
        self.max_speed_threshold = 50.0  # Acima desta velocidade, o tremor é considerado máximo.
        
        self.min_pull = 0.05  # Força mínima do olhar (permite controle fino do mouse)
        self.max_pull = 0.20  # Força máxima do olhar (assume o controle durante tremores fortes)
        
        # Variáveis para cálculo de velocidade
        self.prev_pos = None
        self.smoothed_speed = 0.0
        self.ema_alpha = 0.1 # Fator de suavização para a velocidade

    def update_and_get_pull_strength(self, current_pos):
        current_pos = np.array(current_pos)
        if self.prev_pos is None:
            self.prev_pos = current_pos
            return self.min_pull

        # 1. Medir a velocidade instantânea do mouse
        distance = np.linalg.norm(current_pos - self.prev_pos)
        # A velocidade é a distância movida desde o último frame.
        # Não precisamos dividir pelo tempo (dt) se o framerate for estável.
        raw_speed = distance

        # 2. Suavizar a medição de velocidade para ter um valor estável
        self.smoothed_speed = (1 - self.ema_alpha) * self.smoothed_speed + self.ema_alpha * raw_speed
        
        # Atualiza a posição anterior
        self.prev_pos = current_pos
        
        # 3. Mapear a velocidade suavizada para o peso do olhar (gaze_pull_strength)
        if self.smoothed_speed <= self.min_speed_threshold:
            return self.min_pull
        if self.smoothed_speed >= self.max_speed_threshold:
            return self.max_pull
        
        # Interpolação linear entre os limiares
        ratio = (self.smoothed_speed - self.min_speed_threshold) / (self.max_speed_threshold - self.min_speed_threshold)
        dynamic_pull = self.min_pull + ratio * (self.max_pull - self.min_pull)
        
        return dynamic_pull

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
# Loop principal de captura (COM PESO DINÂMICO E VISUALIZADOR)
# -----------------------------
def gaze_main_loop(gestures, video_capture, screen, screen_width, screen_height, bold_font, n_points=25, context="my_context"):
    clock = pygame.time.Clock()
    running = True
    iterator = 0
    prev_x, prev_y = 0, 0
    capturing_input = False

    point_positions = [(0.3, 0.5),(0.5, 0.25),(0.75, 0.55),(0.9,0.9),(0.5,0.5)]
    click_points = [make_point(pos[0], pos[1], screen_width, screen_height) for pos in point_positions]
    start_point, end_point = click_points[0], click_points[-1]

    eventos_mouse, eventos_gaze, eventos_final = [], [], []

    # Inicialização da lógica
    mouse_smoother = PositionSmoother(window_size=4)
    final_cursor_pos = np.array(pygame.mouse.get_pos(), dtype=float)
    # NOVO: Inicializa o detector de tremor
    tremor_detector = TremorDetector()
    
    # Parâmetro de suavização do movimento final (pode continuar estático)
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
            if event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos; button = event.button
                click_type, _ = detect_click(x, y, button)
                if not capturing_input and click_type == 'double_click' and hypot(x-start_point[0], y-start_point[1]) <= POINT_RADIUS:
                    capturing_input = True; print("Captura iniciada")
                elif capturing_input and click_type == 'double_click' and hypot(x-end_point[0], y-end_point[1]) <= POINT_RADIUS:
                    capturing_input = False; running = False; print("Captura encerrada")
            
            pygame.draw.circle(screen, GREEN, start_point, POINT_RADIUS)
            pygame.draw.circle(screen, RED, end_point, POINT_RADIUS)
            for point in click_points[1:-1]: pygame.draw.circle(screen, BLUE, point, POINT_RADIUS)

            # Lógica de atração com peso dinâmico
            if gaze_event is not None and gaze_event.point is not None:
                mouse_pos = np.array(pygame.mouse.get_pos())
                gaze_pos = np.array(gaze_event.point)
                
                # Obtém o peso do olhar dinamicamente com base no tremor
                gaze_pull_strength = tremor_detector.update_and_get_pull_strength(mouse_pos)
                
                smoothed_mouse_pos = mouse_smoother.smooth(mouse_pos[0], mouse_pos[1])
                attraction_vector = gaze_pos - smoothed_mouse_pos
                target_pos = smoothed_mouse_pos + gaze_pull_strength * attraction_vector
                final_cursor_pos = (1 - cursor_lerp_factor) * final_cursor_pos + cursor_lerp_factor * target_pos

            
            if capturing_input and gaze_event is not None:
                timestamp = pygame.time.get_ticks()
                mouse_x, mouse_y = pygame.mouse.get_pos()
                eventos_mouse.append({'timestamp': timestamp, 'x': mouse_x, 'y': mouse_y})
                eventos_gaze.append({'timestamp': timestamp, 'x': gaze_event.point[0], 'y': gaze_event.point[1]})
                eventos_final.append({'timestamp': timestamp, 'x': final_cursor_pos[0], 'y': final_cursor_pos[1]})

        surface = pygame.display.get_surface()
        if surface and gaze_event is not None and gaze_event.point is not None:
            pygame.draw.circle(surface, RED, gaze_event.point, GAZE_POINT_RADIUS)
            if not calibrate:
                pygame.draw.circle(surface, CYAN, (int(final_cursor_pos[0]), int(final_cursor_pos[1])), 15)

        # Visualizador de depuração para o peso dinâmico
        if not calibrate:
            # Barra de fundo
            pygame.draw.rect(screen, (50, 50, 50), [10, screen_height - 40, 200, 30])
            # Barra de tremor (velocidade)
            speed_ratio = min(tremor_detector.smoothed_speed / tremor_detector.max_speed_threshold, 1.0)
            pygame.draw.rect(screen, YELLOW, [10, screen_height - 40, 200 * speed_ratio, 15])
            # Barra de força do olhar (pull strength)
            pull_ratio = (gaze_pull_strength - tremor_detector.min_pull) / (tremor_detector.max_pull - tremor_detector.min_pull)
            pygame.draw.rect(screen, (0, 150, 255), [10, screen_height - 25, 200 * pull_ratio, 15])

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