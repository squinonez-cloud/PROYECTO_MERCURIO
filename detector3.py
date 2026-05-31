import cv2
import mediapipe as mp
import face_recognition
import os

MAX_CARAS = 4  # Número máximo de rostros a detectar
MAX_MANOS = 8  # Número máximo de manos a detectar
# ========================================================

# Diccionario de características
caracteristicas = {
    "Jonathan Valenzuela": {"Edad": 16, "Nacionalidad": "guatemalteco", "Altura": "1.64m"},
    "Alexandre Orozco": {"Edad": 16, "Nacionalidad": "guatemalteco", "Altura": "1.66m"},
    "Santiago Quiñones": {"Edad": 16, "Nacionalidad": "guatemalteco", "Altura": "1.70m"},
    "Dylan Godoy": {"Edad": 16, "Nacionalidad": "guatemalteco", "Altura": "1.68m"}
}

# Cargar rostros conocidos desde carpeta "caritas"
nombres = []
codigos = []

for nombre in os.listdir("caritas/"):
    ruta = f"caritas/{nombre}"
    if not os.path.isdir(ruta):
        continue
    for archivo in os.listdir(ruta):
        img = face_recognition.load_image_file(f"{ruta}/{archivo}")
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            codigos.append(enc[0])
            nombres.append(nombre)

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
def dibujar_enmarcado(frame, x, y, w, h, color, etiqueta):
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, etiqueta, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

    # --- Reconocimiento facial con face_recognition ---
    locs = face_recognition.face_locations(frame_rgb)
    encs = face_recognition.face_encodings(frame_rgb, locs)

    for (top, right, bottom, left), enc in zip(locs, encs):
        matches = face_recognition.compare_faces(codigos, enc)
        nombre = "Desconocido"

        if True in matches:
            idx = matches.index(True)
            nombre = nombres[idx]

        # Dibujar cuadro verde con nombre
        dibujar_enmarcado(frame, left, top, right-left, bottom-top, (0,255,0), nombre)

        # Mostrar características debajo del cuadro
        if nombre in caracteristicas:
            datos = caracteristicas[nombre]
            y_offset = bottom + 20
            cv2.putText(frame, f"Edad: {datos['Edad']}", (left, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Nacionalidad: {datos['Nacionalidad']}", (left, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Altura: {datos['Altura']}", (left, y_offset+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # --- PROCESAR MANOS ---
    resultados_manos = modelo_manos.process(frame_rgb)
    if resultados_manos.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(resultados_manos.multi_hand_landmarks):
            # Mantener colores originales: puntos rojos y conexiones blancas
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Dibujar cuadro azul para cada mano
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)
            cv2.putText(frame, f"Mano {i+1}", (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

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
