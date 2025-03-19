############################################################
### Comparando as tecnologias de Haarcascade x HOG x CNN ###
############################################################

import cv2 as cv
import dlib as dl

## HOG ##
imagem_hog = cv.imread("./resources/Images/people3.jpg")
detector_face_hog = dl.get_frontal_face_detector()
detection_hog = detector_face_hog(imagem_hog, 4)  # O valor numérico é relativo a escala

for face in detection_hog:
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(imagem_hog, (l, t), (r, b), (0, 255, 0), 2)

# HOG
print(cv.imshow("color image", imagem_hog))

# # Waits for a keystroke
cv.waitKey(0)
