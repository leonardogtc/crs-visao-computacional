import cv2 as cv
import dlib as dl

# Carregar a imagem com OpenCV
imagem = cv.imread("./resources/Images/people2.jpg")

detector_face_hog = dl.get_frontal_face_detector()

detection = detector_face_hog(imagem, 1)  # O valor numérico é relativo a escala

print(detection)

for face in detection:
    # print(face)
    # print(face.left())
    # print(face.top())
    # print(face.right())
    # print(face.bottom())
    l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
    cv.rectangle(imagem, (l, t), (r, b), (0, 255, 0), 2)


# Carrega a imagem com os retângulos de detecção facial
print(cv.imshow("color image", imagem))

# Waits for a keystroke
cv.waitKey(0)
