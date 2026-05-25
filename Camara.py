import cv2
import mediapipe as mp

# 1. Inicializar MediaPipe Holistic y las herramientas de dibujo
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Configuración del modelo integral (Rostro + Manos + Postura)
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Función auxiliar para calcular y dibujar el cuadro delimitador (cuadrado)
def dibujar_cuadro(frame, landmarks, color, etiqueta):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    # Encontrar los puntos extremos de la parte del cuerpo
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        if cx < x_min: x_min = cx
        if cx > x_max: x_max = cx
        if cy < y_min: y_min = cy
        if cy > y_max: y_max = cy

    # Añadir un pequeño margen al cuadro
    margen = 15
    x_min = max(0, x_min - margen)
    y_min = max(0, y_min - margen)
    x_max = min(w, x_max + margen)
    y_max = min(h, y_max + margen)

    # Dibujar el rectángulo y el texto de la etiqueta
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(frame, etiqueta, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 3. Encender la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Efecto espejo
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar todo el cuerpo
    resultados = holistic.process(frame_rgb)

    # --- DETECCIÓN Y CUADRO DEL ROSTRO ---
    if resultados.face_landmarks:
        # Dibujar malla facial opcional (puedes comentarla si solo quieres el cuadro)
        mp_drawing.draw_landmarks(
            frame, resultados.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
        )
        # Dibujar cuadro verde para el Rostro
        dibujar_cuadro(frame, resultados.face_landmarks, (0, 255, 0), "Rostro")

    # --- DETECCIÓN Y CUADRO DE MANO IZQUIERDA ---
    if resultados.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Dibujar cuadro cian para mano izquierda
        dibujar_cuadro(frame, resultados.left_hand_landmarks, (255, 255, 0), "Mano Derecha")

    # --- DETECCIÓN Y CUADRO DE MANO DERECHA ---
    if resultados.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Dibujar cuadro azul para mano derecha
        dibujar_cuadro(frame, resultados.right_hand_landmarks, (255, 0, 0), "Mano Izquierda")

    # Mostrar la ventana
    cv2.imshow('Deteccion Multiple con Cuadros', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
holistic.close()
cv2.destroyAllWindows()