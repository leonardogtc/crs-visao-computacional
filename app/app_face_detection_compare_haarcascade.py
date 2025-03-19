############################################################
### Comparando as tecnologias de Haarcascade x HOG x CNN ###
############################################################

import cv2 as cv
import dlib as dl

## Haarcascades ##
imagem = cv.imread("./resources/Images/people3.jpg")
imagem_cinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
detection_haarcascade = detector_facial = cv.CascadeClassifier(
    "./resources/Cascades/haarcascade_frontalface_default.xml"
)
detection001 = detection_haarcascade.detectMultiScale(
    imagem_cinza, scaleFactor=1.001, minNeighbors=5, minSize=(5, 5)
)

print(detection001)

# Colocando retângulos posicionados nas faces através do array de posições
for x, y, w, h in detection001:
    cv.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

# HAARCASCADE
print(cv.imshow("color image", imagem))

# # Waits for a keystroke
cv.waitKey(0)

## HOG ##
imagem_hog = cv.imread("./resources/Images/people3.jpg")
detector_face_hog = dl.get_frontal_face_detector()
detection_hog = detector_face_hog(imagem_hog, 1)  # O valor numérico é relativo a escala

for face in detection_hog:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(imagem_hog, (l, t), (r, b), (0, 255, 0), 2)

# HOG
print(cv.imshow("color image", imagem_hog))

# # Waits for a keystroke
cv.waitKey(0)
