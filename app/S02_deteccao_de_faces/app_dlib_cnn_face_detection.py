################################################################
### Detecção de Faces com CNN (Redes Neurais Convolucionais) ###
################################################################

# Carrega bibliotecas
import cv2 as cv
import dlib as dl

# Carregar a imagem com OpenCV
imagem = cv.imread("./resources/Images/people2.jpg")

detector_face_cnn = dl.cnn_face_detection_model_v1(
    "./resources/Weights/mmod_human_face_detector.dat"
)

detections = detector_face_cnn(imagem, 1)

for face in detections:
    l, t, r, b, c = (
        face.rect.left(),
        face.rect.top(),
        face.rect.right(),
        face.rect.bottom(),
        face.confidence,
    )
    cv.rectangle(imagem, (l, t), (r, b), (0, 255, 255), 2)

# Carrega a imagem com os retângulos de detecção facial
print(cv.imshow("color image", imagem))

# Waits for a keystroke
cv.waitKey(0)
