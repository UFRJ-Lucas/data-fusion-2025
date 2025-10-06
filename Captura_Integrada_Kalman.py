# VERSÃO COM FILTRO DE KALMAN

# Bibliotecas
import pygame, time, cv2, random, math
import numpy as np
import pandas as pd
from math import hypot
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3
from filterpy.kalman import KalmanFilter # NOVO: Importa o Filtro de Kalman

# --- Parâmetros e Constantes ---
DOUBLE_CLICK_MAX_INTERVAL = 0.3
POINT_RADIUS = 20
GAZE_POINT_RADIUS = 30
eventos = []
ultimo_click = {}
RED = (255, 0, 100); BLUE = (100, 0, 255); GREEN = (0, 255, 0); BLANK = (125,125,125)
WHITE = (255, 255, 255); CYAN = (0, 255, 255); YELLOW = (255, 255, 0)


# --- Classes Auxiliares ---

# NOVO: Classe que encapsula a lógica do Filtro de Kalman
class KalmanStabilizer:
    def __init__(self, process_noise, mouse_noise_normal, mouse_noise_tremor):
        # O estado do filtro é [posição_x, posição_y, velocidade_x, velocidade_y]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # Matriz de transição de estado (F): descreve a física do movimento
        # pos_nova = pos_antiga + vel * dt. Assumindo dt=1 frame.
        self.kf.F = np.array([[1, 0, 1, 0],   # x = x + vx
                               [0, 1, 0, 1],   # y = y + vy
                               [0, 0, 1, 0],   # vx = vx
                               [0, 0, 0, 1]])  # vy = vy

        # Matriz de medição (H): mapeia o estado para a medição
        # Nós medimos apenas a posição (x, y).
        self.kf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])

        # Covariância do ruído do processo (Q): "Incerteza do modelo de movimento"
        # Quão bruscamente o usuário pode mudar de direção?
        self.kf.Q *= process_noise

        # Covariância da incerteza inicial (P)
        self.kf.P *= 10

        # Covariância do ruído da medição (R): "Confiança no sensor"
        # Teremos dois níveis de confiança no mouse.
        self.R_normal = np.eye(2) * mouse_noise_normal
        self.R_tremor = np.eye(2) * mouse_noise_tremor
        self.kf.R = self.R_normal

    def start(self, initial_pos):
        """Inicializa o estado do filtro com a primeira posição do mouse."""
        self.kf.x = np.array([initial_pos[0], initial_pos[1], 0, 0])

    def update(self, mouse_pos, is_tremor):
        """Executa um ciclo de predição e atualização do filtro."""
        # Decide o quão confiável é a medição do mouse neste frame
        if is_tremor:
            self.kf.R = self.R_tremor # Se há tremor, desconfie muito do mouse
        else:
            self.kf.R = self.R_normal # Em movimento normal, confie mais no mouse

        self.kf.predict()
        self.kf.update(mouse_pos)
        
        # Retorna a posição estimada [x, y] do estado do filtro
        return self.kf.x[0:2]


class TremorSimulator:
    def __init__(self, chance=0.2, amplitude=15, frequency=10, duration=0.5): self.chance_per_frame=chance/60; self.amplitude=amplitude; self.frequency=frequency; self.duration_frames=duration*60; self.is_tremoring=False; self.tremor_frames_left=0; self.tremor_start_time=0
    def update(self, true_mouse_pos):
        now=time.time()
        if not self.is_tremoring and random.random()<self.chance_per_frame: self.is_tremoring=True; self.tremor_frames_left=self.duration_frames; self.tremor_start_time=now
        if self.is_tremoring:
            self.tremor_frames_left-=1
            if self.tremor_frames_left<=0: self.is_tremoring=False; return true_mouse_pos
            elapsed_time=now-self.tremor_start_time; offset_x=self.amplitude*math.sin(elapsed_time*self.frequency); offset_y=self.amplitude*math.cos(elapsed_time*self.frequency*1.5)
            return (true_mouse_pos[0]+offset_x, true_mouse_pos[1]+offset_y)
        else: return true_mouse_pos


# --- Funções de Setup ---
# (register_event, detect_click, init_gaze_screen, setup_calibration, make_point não mudam)
def register_event(x,y,event_type,button,click_count): eventos.append({'timestamp':time.time(),'x':x,'y':y,'event_type':event_type,'button':button,'click_count':click_count})
def detect_click(x,y,button):
    global ultimo_click; now=time.time(); key=str(button)
    if key in ultimo_click:
        dt=now-ultimo_click[key]['timestamp']; dist=hypot(x-ultimo_click[key]['x'],y-ultimo_click[key]['y'])
        if dt<=DOUBLE_CLICK_MAX_INTERVAL and dist<=POINT_RADIUS: ultimo_click[key]={'timestamp':now,'x':x,'y':y}; register_event(x,y,'double_click',button,2); return 'double_click',2
    ultimo_click[key]={'timestamp':now,'x':x,'y':y}; register_event(x,y,'click',button,1); return 'click',1
def init_gaze_screen(scale=0.8,font_size=48,calibration_radius=300):
    pygame.init(); pygame.font.init(); screen_info=pygame.display.Info(); w,h=int(scale*screen_info.current_w),int(scale*screen_info.current_h)
    screen=pygame.display.set_mode((w,h)); pygame.display.set_caption("Integração Mouse + Câmera"); bold_font=pygame.font.Font(None,font_size); bold_font.set_bold(True)
    gestures=EyeGestures_v3(calibration_radius=calibration_radius); video_capture=VideoCapture(0); return gestures,video_capture,screen,w,h,bold_font
def setup_calibration(gestures,max_points=25,context="my_context"):
    x,y=np.arange(0,1.1,0.2),np.arange(0,1.1,0.2); xx,yy=np.meshgrid(x,y); calibration_map=np.column_stack([xx.ravel(),yy.ravel()])
    n_points=min(len(calibration_map),max_points); np.random.shuffle(calibration_map); gestures.uploadCalibrationMap(calibration_map,context=context); gestures.setFixation(1.0); return n_points
def make_point(x_percent,y_percent,screen_width,screen_height): return (x_percent*screen_width,y_percent*screen_height)


# --- Loop Principal de Captura ---
def gaze_main_loop(gestures, video_capture, screen, screen_width, screen_height, bold_font, n_points=25, context="my_context"):
    clock = pygame.time.Clock()
    running = True; iterator = 0; prev_x, prev_y = 0, 0; capturing_input = False

    point_positions = [(0.25, 0.75),(0.3, 0.25),(0.70, 0.25), (0.75, 0.6),(0.5,0.5)]
    click_points = [make_point(pos[0], pos[1], screen_width, screen_height) for pos in point_positions]
    start_point, end_point = click_points[0], click_points[-1]

    eventos_mouse_real, eventos_mouse_com_tremor, eventos_gaze, eventos_final = [], [], [], []

    # --- PARÂMETROS DE AJUSTE (FILTRO DE KALMAN) ---
    p_chance_tremor = 1.5; 
    p_amplitude_tremor = 25
    
    # Parâmetros de detecção (continuam os mesmos)
    p_mouse_speed_threshold = 15.0
    p_gaze_speed_threshold = 10.0

    # Parâmetros do Filtro de Kalman
    p_process_noise = 0.1       # Agilidade: quão rápido o usuário pode mudar de direção?
    p_mouse_noise_normal = 5.0  # Confiança no mouse em movimento normal
    p_mouse_noise_tremor = 500.0 # Desconfiança no mouse durante um tremor (valor alto!)
    # ---------------------------------------------------

    tremor_simulator = TremorSimulator(chance=p_chance_tremor, amplitude=p_amplitude_tremor, frequency=30, duration=0.6)
    
    # NOVO: Inicialização do estabilizador Kalman
    stabilizer = KalmanStabilizer(p_process_noise, p_mouse_noise_normal, p_mouse_noise_tremor)
    stabilizer.start(pygame.mouse.get_pos())
    
    prev_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
    prev_gaze_pos = None

    pygame.mouse.set_visible(True)

    while running:
        # (Loop de eventos sem alterações)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
            if not (iterator <= n_points) and event.type == pygame.MOUSEBUTTONUP:
                x, y = event.pos; button = event.button; click_type, _ = detect_click(x, y, button)
                if not capturing_input and click_type == 'double_click' and hypot(x - start_point[0], y - start_point[1]) <= POINT_RADIUS:
                    capturing_input = True; print("Captura iniciada")
                elif capturing_input and click_type == 'double_click' and hypot(x - end_point[0], y - end_point[1]) <= POINT_RADIUS:
                    capturing_input = False; running = False; print("Captura encerrada")

        if not running: break
        
        ret, frame = video_capture.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = np.flip(frame, axis=1)
        
        calibrate = (iterator <= n_points)
        try:
            gaze_event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context=context)
        except TypeError:
            gaze_event, calibration = None, None
        if gaze_event is None and calibration is None: continue
        
        screen.fill(BLANK)
        if gaze_event and gaze_event.sub_frame is not None: screen.blit(pygame.surfarray.make_surface(np.rot90(gaze_event.sub_frame)), (0, 0))
        
        if calibrate and calibration is not None:
            if calibration.point[0] != prev_x or calibration.point[1] != prev_y: iterator += 1; prev_x, prev_y = calibration.point[0], calibration.point[1]
            pygame.draw.circle(screen, BLUE, calibration.point, calibration.acceptance_radius)
            text_surface = bold_font.render(f"{iterator}/{n_points}", True, WHITE); text_square = text_surface.get_rect(center=calibration.point); screen.blit(text_surface, text_square)
        
        elif not calibrate:
            mouse_pos_real = np.array(pygame.mouse.get_pos())
            mouse_pos_com_tremor = mouse_pos_real
            if capturing_input:
                pygame.mouse.set_visible(False); mouse_pos_com_tremor = np.array(tremor_simulator.update(mouse_pos_real))
            else:
                pygame.mouse.set_visible(True); stabilizer.start(mouse_pos_real) # Reinicia o filtro

            # LÓGICA DE ESTABILIZAÇÃO COM KALMAN
            is_tremor_detected = False
            if gaze_event is not None and gaze_event.point is not None:
                gaze_pos = np.array(gaze_event.point)
                mouse_speed = np.linalg.norm(mouse_pos_com_tremor - prev_mouse_pos)
                gaze_speed = np.linalg.norm(gaze_pos - prev_gaze_pos) if prev_gaze_pos is not None else 0
                
                if mouse_speed > p_mouse_speed_threshold and gaze_speed < p_gaze_speed_threshold:
                    is_tremor_detected = True
                
                prev_gaze_pos = gaze_pos
            
            final_cursor_pos = stabilizer.update(mouse_pos_com_tremor, is_tremor_detected)
            prev_mouse_pos = mouse_pos_com_tremor

            if capturing_input:
                timestamp = pygame.time.get_ticks()
                eventos_mouse_real.append({'timestamp': timestamp, 'x': mouse_pos_real[0], 'y': mouse_pos_real[1]})
                eventos_mouse_com_tremor.append({'timestamp': timestamp, 'x': mouse_pos_com_tremor[0], 'y': mouse_pos_com_tremor[1]})
                eventos_final.append({'timestamp': timestamp, 'x': final_cursor_pos[0], 'y': final_cursor_pos[1]})
                if gaze_event: eventos_gaze.append({'timestamp': timestamp, 'x': gaze_event.point[0], 'y': gaze_event.point[1]})

            # Desenho da UI
            pygame.draw.circle(screen, GREEN, start_point, POINT_RADIUS); pygame.draw.circle(screen, RED, end_point, POINT_RADIUS)
            for point in click_points[1:-1]: pygame.draw.circle(screen, BLUE, point, POINT_RADIUS)
            status_text = "Gravação iniciada..." if capturing_input else "Clique duplo no alvo VERDE para iniciar a gravação"
            text_color = GREEN if capturing_input else WHITE
            text_surface = bold_font.render(status_text, True, text_color); text_rect = text_surface.get_rect(center=(screen_width//2, 40)); screen.blit(text_surface, text_rect)
            pygame.draw.circle(screen, BLUE, (int(mouse_pos_com_tremor[0]), int(mouse_pos_com_tremor[1])), 10)
            pygame.draw.circle(screen, CYAN, (int(final_cursor_pos[0]), int(final_cursor_pos[1])), 15)

        if gaze_event is not None and gaze_event.point is not None:
            pygame.draw.circle(screen, RED, gaze_event.point, GAZE_POINT_RADIUS)
            
        pygame.display.flip()
        clock.tick(60)
        
    pygame.mouse.set_visible(True)
    
    # Salva apenas os dataframes relevantes
    df_mouse_real = pd.DataFrame(eventos_mouse_real)
    df_mouse_com_tremor = pd.DataFrame(eventos_mouse_com_tremor)
    df_gaze = pd.DataFrame(eventos_gaze)
    df_final = pd.DataFrame(eventos_final)
    print(f'Registros: Real={len(df_mouse_real)}, Tremor={len(df_mouse_com_tremor)}, Gaze={len(df_gaze)}, Final={len(df_final)}')
    df_mouse_real.to_pickle("df_mouse_real_kalman.pkl")
    df_mouse_com_tremor.to_pickle("df_mouse_com_tremor_kalman.pkl")
    df_gaze.to_pickle("df_gaze_original_kalman.pkl")
    df_final.to_pickle("df_cursor_final_kalman.pkl")


# --- Funções Finais ---
def finalize_gaze(video_capture):
    pygame.mouse.set_visible(True); pygame.quit()
    if video_capture and video_capture.cap: video_capture.cap.release(); del video_capture

if __name__ == "__main__":
    gestures, video_capture, screen, w, h, bold_font = init_gaze_screen(scale=1)
    n_points = setup_calibration(gestures, max_points=25)
    gaze_main_loop(gestures, video_capture, screen, w, h, bold_font, n_points=n_points)
    finalize_gaze(video_capture)