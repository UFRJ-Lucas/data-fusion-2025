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

# Offsets dos pontos
x_offset_inicio, y_offset_inicio = -150, 150
x_offset_intermediario, y_offset_intermediario = 0, -120
x_offset_fim, y_offset_fim = 0, 50

# Variáveis globais
eventos = []
ultimo_click = {}
capturando = False

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
def setup_calibration(gestures, n_points=25, context="my_context"):
    x = np.arange(0, 1.1, 0.2)
    y = np.arange(0, 1.1, 0.2)
    xx, yy = np.meshgrid(x, y)
    
    calibration_map = np.column_stack([xx.ravel(), yy.ravel()])
    n_points = min(len(calibration_map), n_points)
    np.random.shuffle(calibration_map)
    
    gestures.uploadCalibrationMap(calibration_map, context=context)
    gestures.setFixation(1.0)
    
    return n_points


# -----------------------------
# Capturar trajetórias
# -----------------------------
def capturar_trajetoria_v2(gestures, video_capture, WIDTH, HEIGHT):
    """
    Captura simultaneamente eventos do mouse e posições do eye tracker.
    Retorna dois dataframes: df_mouse e df_gaze.
    """
    global capturando

    # Pontos
    ponto_inicio = (WIDTH // 4 + x_offset_inicio, HEIGHT // 2 + y_offset_inicio)
    ponto_intermediario = (WIDTH // 2 + x_offset_intermediario, HEIGHT // 2 + y_offset_intermediario)
    ponto_fim = (3 * WIDTH // 4 + x_offset_fim, HEIGHT // 2 + y_offset_fim)

    # DataFrames
    eventos_mouse = []
    eventos_gaze = []

    rodando = True
    while rodando:
        #screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (0, 255, 0), ponto_inicio, POINT_RADIUS)
        pygame.draw.circle(screen, (0, 0, 255), ponto_intermediario, POINT_RADIUS)
        pygame.draw.circle(screen, (255, 0, 0), ponto_fim, POINT_RADIUS)
        pygame.display.flip()


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rodando = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    rodando = False
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos
                button = event.button
                click_type, click_count = detect_click(x, y, button)

                if not capturando and click_type == 'double_click' and hypot(x - ponto_inicio[0], y - ponto_inicio[1]) <= POINT_RADIUS:
                    capturando = True
                    print("Captura iniciada")
                elif capturando and click_type == 'double_click' and hypot(x - ponto_fim[0], y - ponto_fim[1]) <= POINT_RADIUS:
                    capturando = False
                    rodando = False
                    print("Captura encerrada")

            elif event.type == pygame.MOUSEMOTION and capturando:
                x, y = event.pos
                eventos_mouse.append({
                    'timestamp': time.time(),
                    'x': x,
                    'y': y,
                    'event_type': 'move',
                    'button': 'mouse',
                    'click_count': 0
                })

                # Captura do frame do eye tracker
                ret, frame = video_capture.read()
                if ret:
                    gaze_event, _ = gestures.step(frame, False, WIDTH, HEIGHT, "my_context")
                    if capturando and gaze_event is not None:
                        eventos_gaze.append({
                            'timestamp': time.time(),
                            'x': gaze_event.point[0],
                            'y': gaze_event.point[1],
                            'fixation': gaze_event.fixation,
                            'saccades': gaze_event.saccades
                        })
    pygame.quit()
    df_mouse = pd.DataFrame(eventos_mouse)
    df_gaze = pd.DataFrame(eventos_gaze)
    print(f'Registros capturados do mouse: {len(df_mouse)}')
    print(f'Registros capturados do gaze: {len(df_gaze)}')
    return df_mouse, df_gaze

# -----------------------------
# Loop principal de captura
# -----------------------------
def gaze_main_loop(gestures, video_capture, screen, screen_width, screen_height, bold_font, n_points=25, context="my_context"):
    clock = pygame.time.Clock()
    running = True
    iterator = 0
    prev_x, prev_y = 0, 0

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
        if not ret:
            frame.release()
            cam_frame.realease()
            running = False  # Encerrar loop imediatamente se falha na captura
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1)
        
        calibrate = (iterator <= n_points)
        gaze_event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context=context)
        
        if gaze_event is None and calibration is None:
            continue
        
        screen.fill((0,0,0))
        
        # Mostrar câmera reduzida no canto
        if gaze_event is not None and hasattr(gaze_event, 'sub_frame'):
            cam_frame = cv2.resize(gaze_event.sub_frame, (200,150))
        else:
            cam_frame = np.zeros((150,200,3), dtype=np.uint8)
        cam_surf = pygame.surfarray.make_surface(np.rot90(cam_frame))
        screen.blit(cam_surf, (0,0))
        
        # Visualização da calibração
        if calibrate and calibration is not None:
            if calibration.point[0] != prev_x or calibration.point[1] != prev_y:
                iterator += 1
                prev_x, prev_y = calibration.point
            calibration.acceptance_radius = 50
            pygame.draw.circle(screen, (0,0,255), calibration.point, calibration.acceptance_radius)
            text_surface = bold_font.render(f"{iterator}/{n_points}", True, (255,255,255))
            text_square = text_surface.get_rect(center=calibration.point)
            screen.blit(text_surface, text_square)
        
        # Após calibração
        if not calibrate:
            df_mouse, df_eye = capturar_trajetoria_v2(gestures, video_capture, screen_width, screen_height)

            # Salvar DataFrames
            df_mouse.to_pickle("df_mouse_original.pkl")
            df_eye.to_pickle("df_gaze_original.pkl")

        surface = pygame.display.get_surface()
        if surface:
        # Visualização do algoritmo
            if gaze_event is not None:
                algo = gestures.whichAlgorithm(context=context)
                color_map = {"Ridge": (255,0,100), "LassoCV": (100,0,255)}
                color = color_map.get(algo, (0,255,0))
                pygame.draw.circle(surface, color, gaze_event.point, 50)
                if gaze_event.saccades:
                    pygame.draw.circle(surface, (0,255,0), gaze_event.point, 50)
                
                font = pygame.font.SysFont('Comic Sans MS', 30)
                text_surface = font.render(algo, False, (0,0,0))
                surface.blit(text_surface, gaze_event.point)
            
            pygame.display.flip()
            clock.tick(60)
        else:
            running = False
            break

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
    n_points = setup_calibration(gestures, n_points=25)
    gaze_main_loop(gestures, video_capture, screen, w, h, bold_font, n_points=n_points)
    finalize_gaze(video_capture)