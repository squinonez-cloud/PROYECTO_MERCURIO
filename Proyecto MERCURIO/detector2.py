import cv2
import os
import numpy as np
import mediapipe as mp

MAX_CARAS = 4
MAX_MANOS = 8

# ========================================================
# Lista de nombres según las subcarpetas en "caritas"
nombres = ["Jonathan Valenzuela", "Alexandre Orozco", "Santiago Quiñones", "Dylan Godoy"]

# Diccionario con características extra
caracteristicas = {
    "Jonathan Valenzuela": {"nacionalidad": "Guatemala", "altura": "1.64m", "caracteristica": "Colocho"},
    "Alexandre Orozco": {"nacionalidad": "México", "altura": "1.66", "caracteristica": "Cabello corto"},
    "Santiago Quiñones": {"nacionalidad": "Colombia", "altura": "1.70m", "caracteristica": "Barba"},
    "Dylan Godoy": {"nacionalidad": "Argentina", "altura": "1.68m", "caracteristica": "Cabello largo"}
}

# ========================================================
# ENTRENAMIENTO LBPH
imagenes, etiquetas = [], []

for i, nombre in enumerate(nombres):
    carpeta = f"caritas/{nombre}"
    for archivo in os.listdir(carpeta):
        img = cv2.imread(os.path.join(carpeta, archivo), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        imagenes.append(img)
        etiquetas.append(i)

imagenes = np.array(imagenes)
etiquetas = np.array(etiquetas)

reconocedor = cv2.face.LBPHFaceRecognizer_create()
reconocedor.train(imagenes, etiquetas)

# ========================================================
# Inicializar herramientas de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

modelo_caras = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=MAX_CARAS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

modelo_manos = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_MANOS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def dibujar_enmarcado(frame, x, y, w, h, color, etiqueta):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, etiqueta, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ========================================================
cap = cv2.VideoCapture(0)
print(f"Cámara lista. Detectando hasta {MAX_CARAS} caras y {MAX_MANOS} manos. Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resultados_caras = modelo_caras.process(frame_rgb)
    resultados_manos = modelo_manos.process(frame_rgb)

    # --- PROCESAR ROSTROS ---
    rostros = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gris, 1.3, 5)
    for (x,y,w,h) in rostros:
        roi = gris[y:y+h, x:x+w]
        id_pred, conf = reconocedor.predict(roi)

        if conf < 90:  # puedes ajustar este umbral
            nombre = nombres[id_pred]
            datos = caracteristicas[nombre]
            etiqueta = f"{nombre} - {datos['nacionalidad']} ({datos['caracteristica']}, {datos['altura']})"
        else:
            etiqueta = "Desconocido"

        dibujar_enmarcado(frame, x, y, w, h, (0,255,0), etiqueta)

    # --- PROCESAR MANOS ---
    if resultados_manos.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(resultados_manos.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            dibujar_enmarcado(frame, x_min, y_min, x_max-x_min, y_max-y_min, (255,0,0), f"Mano {i+1}")

    cv2.imshow('Deteccion Multiple de Caras y Manos + Reconocimiento', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
modelo_caras.close()
modelo_manos.close()
cv2.destroyAllWindows()
