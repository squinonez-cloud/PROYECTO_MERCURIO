import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# Abrir cámara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = camara.read()

    if not ret:
        print("No se pudo recibir el video")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resultado = hands.process(rgb)

    if resultado.multi_hand_landmarks:

        for hand_landmarks in resultado.multi_hand_landmarks:

            color_azul = (255, 0, 0)
            color_blanco = (255, 255, 255)

            # Dibujar landmarks y conexiones
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,

                # Puntos
                mp_draw.DrawingSpec(
                    color=color_blanco,
                    thickness=2,
                    circle_radius=3
                ),

                # Líneas
                mp_draw.DrawingSpec(
                    color=color_azul,
                    thickness=2
                )
            )

    cv2.imshow("Deteccion de Manos", frame)

    if cv2.waitKey(1) == 27:
        break

camara.release()
cv2.destroyAllWindows()