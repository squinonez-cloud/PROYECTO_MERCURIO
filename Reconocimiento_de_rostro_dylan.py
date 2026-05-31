import cv2
import mediapipe as mp

# ===================== DATOS DE LOS 4 ESTUDIANTES =====================
estudiantes = [
    {"nombre": "Dylan Godoy", "edad": 17, "pais": "Guatemala", "altura": "1.70 m"},
    {"nombre": "Alexandre Orozco", "edad": 16, "pais": "Guatemala", "altura": "1.66 m"},
    {"nombre": "Santiago Quiñonez", "edad": 17, "pais": "Guatemala", "altura": "1.78 m"},
    {"nombre": "Jonathan Valenzuela", "edad": 18, "pais": "Guatemala", "altura": "1.65 m"},
]
# ===================== MEDIAPIPE =====================
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== CAMARA =====================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- MODO ESPEJO ----
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb)
    mesh_results = face_mesh.process(rgb)
    hands_results = hands.process(rgb)

    # ===================== CARAS + DATOS =====================
    if face_results.detections:
        for i, detection in enumerate(face_results.detections[:4]):
            estudiante = estudiantes[i]

            box = detection.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            cv2.putText(frame, f"Nombre: {estudiante['nombre']}", (x, y - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"Edad: {estudiante['edad']}", (x, y - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"Pais: {estudiante['pais']}", (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"Altura: {estudiante['altura']}", (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # ===================== PUNTOS EN EL ROSTRO =====================
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 128, 255), thickness=1)
            )

    # ===================== MANOS =====================
    if hands_results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            label = hand_info.classification[0].label
            texto = "MANO IZQUIERDA" if label == "Left" else "MANO DERECHA"
            color = (255, 0, 0) if label == "Left" else (0, 0, 255)

            x_hand = int(hand_landmarks.landmark[0].x * w)
            y_hand = int(hand_landmarks.landmark[0].y * h)

            cv2.putText(frame, texto, (x_hand - 40, y_hand - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ===================== MOSTRAR =====================
    cv2.imshow("Sistema de Reconocimiento - Proyecto Mercurio", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()