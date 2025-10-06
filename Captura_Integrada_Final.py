# VERSÃO UNIFICADA: FREIO ADAPTATIVO + FILTRO DE KALMAN

# Bibliotecas
import pygame, time, cv2, random, math
import numpy as np
import pandas as pd
from math import hypot
from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3
from filterpy.kalman import KalmanFilter

# --- Parâmetros e Constantes ---
DOUBLE_CLICK_MAX_INTERVAL = 0.3; 
POINT_RADIUS = 20; 
GAZE_POINT_RADIUS = 30

eventos = []; 
ultimo_click = {}

# Constantes de cores
RED = (255,0,100)
BLUE = (100,0,255)
GREEN = (0,255,0)
BLANK = (125,125,125)
WHITE = (255,255,255)
CYAN = (0,255,255)
YELLOW = (255,255,0)

# --- Classes Auxiliares  ---
class PositionSmoother:
    """Suaviza uma posição 2D usando média móvel."""
    def __init__(self, window_size=5): self.window_size=window_size; self.history_x=[]; self.history_y=[]
    def smooth(self, x, y):
        self.history_x.append(x); self.history_y.append(y)
        if len(self.history_x) > self.window_size: self.history_x.pop(0); self.history_y.pop(0)
        return np.array([sum(self.history_x)/len(self.history_x), sum(self.history_y)/len(self.history_y)])

class KalmanStabilizer:
    def __init__(self, process_noise, mouse_noise_normal, mouse_noise_tremor):
        # O estado do filtro é [posição_x, posição_y, velocidade_x, velocidade_y]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Matriz de transição de estado (F): descreve a física do movimento
        # pos_nova = pos_antiga + vel * dt. Assumindo dt=1 frame.
        self.kf.F = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])

        # Matriz de medição (H): mapeia o estado para a medição
        # Nós medimos apenas a posição (x, y).
        self.kf.H = np.array([[1,0,0,0], [0,1,0,0]])

        # Covariância do ruído do processo (Q): "Incerteza do modelo de movimento"
        # Quão bruscamente o usuário pode mudar de direção?
        self.kf.Q *= process_noise; self.kf.P *= 10

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
        self.kf.R = self.R_tremor if is_tremor else self.R_normal
        self.kf.predict()
        self.kf.update(mouse_pos)
        return self.kf.x[0:2]

class TremorSimulator:
    """Adiciona um tremor simulado a uma trajetória de mouse."""
    def __init__(self, chance=0.2, amplitude=15, frequency=10, duration=0.5): 
        self.chance_per_frame=chance/60
        self.amplitude=amplitude
        self.frequency=frequency
        self.duration_frames=duration*60
        self.is_tremoring=False
        self.tremor_frames_left=0
        self.tremor_start_time=0

    def update(self, true_mouse_pos):
        now=time.time()
        if not self.is_tremoring and random.random()<self.chance_per_frame: 
            self.is_tremoring=True
            self.tremor_frames_left=self.duration_frames
            self.tremor_start_time=now
        if self.is_tremoring:
            self.tremor_frames_left-=1
            if self.tremor_frames_left<=0: 
                self.is_tremoring=False
                return true_mouse_pos
            elapsed_time=now-self.tremor_start_time
            offset_x=self.amplitude*math.sin(elapsed_time*self.frequency)
            offset_y=self.amplitude*math.cos(elapsed_time*self.frequency*1.5)
            return (true_mouse_pos[0]+offset_x, true_mouse_pos[1]+offset_y)
        else: return true_mouse_pos

# --- Funções ---
def register_event(x,y,event_type,button,click_count): 
    eventos.append({'timestamp':time.time(),'x':x,'y':y,'event_type':event_type,'button':button,'click_count':click_count})

def detect_click(x,y,button):
    global ultimo_click; now=time.time(); key=str(button)
    if key in ultimo_click:
        dt=now-ultimo_click[key]['timestamp']; dist=hypot(x-ultimo_click[key]['x'],y-ultimo_click[key]['y'])
        if dt<=DOUBLE_CLICK_MAX_INTERVAL and dist<=POINT_RADIUS: ultimo_click[key]={'timestamp':now,'x':x,'y':y}; register_event(x,y,'double_click',button,2); return 'double_click',2
    ultimo_click[key]={'timestamp':now,'x':x,'y':y}; register_event(x,y,'click',button,1); return 'click',1

def init_gaze_screen(scale=0.8,font_size=48,calibration_radius=300):
    pygame.init(); pygame.font.init()
    screen_info=pygame.display.Info()
    w,h=int(scale*screen_info.current_w),int(scale*screen_info.current_h)
    screen=pygame.display.set_mode((w,h)); pygame.display.set_caption("Integração Mouse + Câmera"); bold_font=pygame.font.Font(None,font_size); bold_font.set_bold(True)
    gestures=EyeGestures_v3(calibration_radius=calibration_radius); video_capture=VideoCapture(0); return gestures,video_capture,screen,w,h,bold_font

def setup_calibration(gestures,max_points=25,context="my_context"):
    x,y=np.arange(0,1.1,0.2),np.arange(0,1.1,0.2)
    xx,yy=np.meshgrid(x,y)
    calibration_map=np.column_stack([xx.ravel(),yy.ravel()])
    n_points=min(len(calibration_map),max_points)
    np.random.shuffle(calibration_map)
    gestures.uploadCalibrationMap(calibration_map,context=context)
    gestures.setFixation(1.0)
    return n_points

def make_point(x_percent,y_percent,screen_width,screen_height): 
    x_pos = x_percent * screen_width
    y_pos = y_percent * screen_height
    return (x_pos, y_pos)

def finalize_gaze(video_capture):
    pygame.mouse.set_visible(True); pygame.quit()
    if video_capture and video_capture.cap: video_capture.cap.release(); del video_capture

# --- Loop Principal ---
def gaze_main_loop(gestures, video_capture, screen, screen_width, screen_height, bold_font, n_points=25, context="my_context"):
    clock = pygame.time.Clock()
    running = True
    iterator = 0
    prev_x, prev_y = 0, 0
    capturing_input = False

    point_positions = [(0.25, 0.75),(0.3, 0.25),(0.70, 0.25),(0.75, 0.6),(0.5,0.5)]
    click_points = [make_point(pos[0], pos[1], screen_width, screen_height) for pos in point_positions]
    start_point, end_point = click_points[0], click_points[-1]

    # --- Listas de eventos para todos os dados ---
    eventos_mouse_real = []
    eventos_mouse_com_tremor = []
    eventos_gaze = []
    eventos_final_freio = [] # Resultado do Filtro "Freio Adaptativo"
    eventos_final_kalman = [] # Resultado do Filtro de Kalman

    # --- PARÂMETROS DE AJUSTE PARA AMBOS OS FILTROS ---
    p_chance_tremor = 1.5; 
    p_amplitude_tremor = 25
    p_mouse_speed_threshold = 15.0  # sensibilidade ao tremor para ativar filtro
    p_gaze_speed_threshold = 10.0   # diferenciar um olhar fixo de olhar em movimento    

    # Parâmetros do "Freio Adaptativo"
    p_suavizacao_mouse_freio = 5    # média das últimas n posições para estabelecer centro estável
    p_lerp_pesado = 0.1             # força do freio pesado, aplicada durante tremor. Move 10% em direção ao alvo (âncora)           
    p_lerp_leve = 0.8               # força do freio leve, aplicada quando sem tremor. Move 60% em direção ao alvo               

    # Parâmetros do Filtro de Kalman
    p_process_noise_kalman = 0.1        # Agilidade do filtro. Quão rápido ele se adapta a mudanças de direção (Baixo = mais suave; Alto = mais responsivo).
    p_mouse_noise_normal_kalman = 5.0   # Confiança no mouse (normal). Quão de perto ele segue a mão (Baixo = mais fiel; Alto = mais suave).
    p_mouse_noise_tremor_kalman = 500.0 # Desconfiança no mouse (tremor). Nível do 'freio' aplicado (Deve ser um valor alto para ignorar o tremor).
    # ---------------------------------------------------

    # --- Inicialização de ambos os filtros ---
    tremor_simulator = TremorSimulator(chance=p_chance_tremor, amplitude=p_amplitude_tremor, frequency=30, duration=0.6)

    # Filtro "Freio Adaptativo"
    mouse_smoother_freio = PositionSmoother(window_size=p_suavizacao_mouse_freio)
    final_cursor_pos_freio = np.array(pygame.mouse.get_pos(), dtype=float)

    # Filtro de Kalman
    stabilizer_kalman = KalmanStabilizer(p_process_noise_kalman, p_mouse_noise_normal_kalman, p_mouse_noise_tremor_kalman)
    stabilizer_kalman.start(pygame.mouse.get_pos())
    
    # Variáveis de estado
    prev_mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
    prev_gaze_pos = None
    pygame.mouse.set_visible(True)

    while running:
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
        try: gaze_event, calibration = gestures.step(frame, calibrate, screen_width, screen_height, context=context)
        except TypeError: gaze_event, calibration = None, None
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
                pygame.mouse.set_visible(True)
                final_cursor_pos_freio = mouse_pos_real # Reseta a posição
                stabilizer_kalman.start(mouse_pos_real) # Reseta o filtro

            is_tremor_detected = False

            if gaze_event is not None and gaze_event.point is not None:
                gaze_pos = np.array(gaze_event.point)
                mouse_speed = np.linalg.norm(mouse_pos_com_tremor - prev_mouse_pos)
                gaze_speed = np.linalg.norm(gaze_pos - prev_gaze_pos) if prev_gaze_pos is not None else 0
                if mouse_speed > p_mouse_speed_threshold and gaze_speed < p_gaze_speed_threshold:
                    is_tremor_detected = True
                prev_gaze_pos = gaze_pos

            # --- Execução dos dois filtros em paralelo ---

            # 1. Filtro "Freio Adaptativo"
            smoothed_mouse_pos_freio = mouse_smoother_freio.smooth(mouse_pos_com_tremor[0], mouse_pos_com_tremor[1])
            cursor_lerp_factor = p_lerp_pesado if is_tremor_detected else p_lerp_leve
            target_pos_freio = smoothed_mouse_pos_freio
            final_cursor_pos_freio = (1 - cursor_lerp_factor) * final_cursor_pos_freio + cursor_lerp_factor * target_pos_freio
            
            # 2. Filtro de Kalman
            final_cursor_pos_kalman = stabilizer_kalman.update(mouse_pos_com_tremor, is_tremor_detected)
            
            prev_mouse_pos = mouse_pos_com_tremor
            
            # Capturar dados
            if capturing_input:
                timestamp = pygame.time.get_ticks()
                eventos_mouse_real.append({'timestamp': timestamp, 'x': mouse_pos_real[0], 'y': mouse_pos_real[1]})
                eventos_mouse_com_tremor.append({'timestamp': timestamp, 'x': mouse_pos_com_tremor[0], 'y': mouse_pos_com_tremor[1]})
                eventos_final_freio.append({'timestamp': timestamp, 'x': final_cursor_pos_freio[0], 'y': final_cursor_pos_freio[1]})
                eventos_final_kalman.append({'timestamp': timestamp, 'x': final_cursor_pos_kalman[0], 'y': final_cursor_pos_kalman[1]})
                if gaze_event: eventos_gaze.append({'timestamp': timestamp, 'x': gaze_event.point[0], 'y': gaze_event.point[1]})

            # Desenho (mostrando apenas o resultado do Kalman para uma tela mais limpa)
            pygame.draw.circle(screen, GREEN, start_point, POINT_RADIUS)
            pygame.draw.circle(screen, RED, end_point, POINT_RADIUS)

            for point in click_points[1:-1]: 
                pygame.draw.circle(screen, BLUE, point, POINT_RADIUS)

            status_text = "Gravação iniciada..." if capturing_input else "Clique duplo no alvo VERDE para iniciar a gravação"

            text_color = GREEN if capturing_input else WHITE

            text_surface = bold_font.render(status_text, True, text_color)
            text_rect = text_surface.get_rect(center=(screen_width//2, 40))
            screen.blit(text_surface, text_rect)

            pygame.draw.circle(screen, BLUE, (int(mouse_pos_com_tremor[0]), int(mouse_pos_com_tremor[1])), 10)
            pygame.draw.circle(screen, CYAN, (int(final_cursor_pos_kalman[0]), int(final_cursor_pos_kalman[1])), 15)

        if gaze_event is not None and gaze_event.point is not None:
            pygame.draw.circle(screen, RED, gaze_event.point, GAZE_POINT_RADIUS)
            
        pygame.display.flip()
        clock.tick(60)
        
    pygame.mouse.set_visible(True)
    
    # --- Salvando todos os resultados ---
    df_mouse_real = pd.DataFrame(eventos_mouse_real)
    df_mouse_com_tremor = pd.DataFrame(eventos_mouse_com_tremor)
    df_gaze = pd.DataFrame(eventos_gaze)
    df_final_freio = pd.DataFrame(eventos_final_freio)
    df_final_kalman = pd.DataFrame(eventos_final_kalman)
    
    print(f'Registros: Real={len(df_mouse_real)}, Tremor={len(df_mouse_com_tremor)}, Gaze={len(df_gaze)}, Final(Freio)={len(df_final_freio)}, Final(Kalman)={len(df_final_kalman)}')

    # Salva os arquivos com nomes distintos
    df_mouse_real.to_pickle("df_mouse_real_unificado.pkl")
    df_mouse_com_tremor.to_pickle("df_mouse_com_tremor_unificado.pkl")
    df_gaze.to_pickle("df_gaze_original_unificado.pkl")
    df_final_freio.to_pickle("df_final_freio_adaptativo.pkl")
    df_final_kalman.to_pickle("df_final_kalman.pkl")

if __name__ == "__main__":
    gestures, video_capture, screen, w, h, bold_font = init_gaze_screen(scale=1)
    n_points = setup_calibration(gestures, max_points=25)
    gaze_main_loop(gestures, video_capture, screen, w, h, bold_font, n_points=n_points)
    finalize_gaze(video_capture)