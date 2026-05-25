import cv2

# 1. Iniciar la captura de video
# El número 0 suele ser la cámara integrada de la laptop. Si tienes otra, prueba con 1 o 2.
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Cámara encendida con éxito. Presiona la tecla 'q' para cerrarla.")

# 2. Bucle para leer la cámara frame por frame
while True:
    # Capturar fotograma por fotograma
    ret, frame = cap.read()

    # Si ret es False, significa que hubo un problema al leer la cámara
    if not ret:
        print("Error: No se puede recibir el feed de la cámara.")
        break

    # 3. Mostrar el video en una ventana interactiva
    cv2.imshow('Mi Camara', frame)

    # 4. Detener el bucle si se presiona la tecla 'q'
    # cv2.waitKey(1) espera 1 milisegundo por una tecla
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Liberar la cámara y cerrar las ventanas de OpenCV
cap.release()
cv2.destroyAllWindows()