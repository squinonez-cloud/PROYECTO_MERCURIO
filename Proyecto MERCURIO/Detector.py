import cv2
import mediapipe as mp

MAX_CARAS = 4  # Número máximo de rostros a detectar
MAX_MANOS = 8  # Número máximo de manos a detectar
# ========================================================

# 1. Inicializar herramientas de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurar el modelo de Rostros (Face Mesh)
modelo_caras = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_CARAS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# Configurar el modelo de Manos
modelo_manos = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_MANOS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Función para calcular y dibujar los cuadros delimitadores
def dibujar_enmarcado(frame, landmarks, color, etiqueta):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    # Encontrar los extremos de los puntos clave detectados
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        if cx < x_min: x_min = cx
        if cx > x_max: x_max = cx
        if cy < y_min: y_min = cy
        if cy > y_max: y_max = cy

    # Añadir un pequeño margen al cuadro para que no quede tan ajustado
    margen = 15
    x_min = max(0, x_min - margen)
    y_min = max(0, y_min - margen)
    x_max = min(w, x_max + margen)
    y_max = min(h, y_max + margen)

    # Dibujar el rectángulo y el texto en la pantalla con OpenCV
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, etiqueta, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 3. Encender la cámara
cap = cv2.VideoCapture(0)

print(f"Cámara lista. Detectando hasta {MAX_CARAS} caras y {MAX_MANOS} manos. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Efecto espejo y conversión de color para MediaPipe
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el fotograma con ambos modelos de IA
    resultados_caras = modelo_caras.process(frame_rgb)
    resultados_manos = modelo_manos.process(frame_rgb)

    # --- PROCESAR ROSTROS ---
    if resultados_caras.multi_face_landmarks:
        for i, face_landmarks in enumerate(resultados_caras.multi_face_landmarks):
            # Dibujamos el cuadro delimitador para cada rostro
            dibujar_enmarcado(frame, face_landmarks, (0, 255, 0), f"Rostro {i+1}")

    # --- PROCESAR MANOS ---
    if resultados_manos.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(resultados_manos.multi_hand_landmarks):
            # Opcional: dibujar las líneas internas de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Dibujamos el cuadro delimitador para cada mano
            dibujar_enmarcado(frame, hand_landmarks, (255, 0, 0), f"Mano {i+1}")

    # Mostrar ventana
    cv2.imshow('Deteccion Multiple de Caras y Manos', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza final
cap.release()
modelo_caras.close()
modelo_manos.close()
cv2.destroyAllWindows()