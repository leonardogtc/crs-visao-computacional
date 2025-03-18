###############################################
# CARREGAMENTO DE IMAGEMS E DETECÇÃO DE FACES #
# #############################################

import cv2 as cv

print(cv.__version__)


# Ler a imagem que será trabalhada
imagem = cv.imread(
    "/home/leonardo/Cursos/crs-visao-computacional-jones/resources/Images/people1.jpg"
)

# print(imagem.shape)
# print(cv.imshow('color image',imagem))

# Redimensionar a imagem
imagem = cv.resize(imagem, (800, 600))

# Converter a imagem para tom de cinza
imagem_cinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

# Detector facial
detector_facial = cv.CascadeClassifier(
    "/home/leonardo/Cursos/crs-visao-computacional-jones/resources/Cascades/haarcascade_frontalface_default.xml"
)
deteccoes = detector_facial.detectMultiScale(imagem_cinza)
print(deteccoes)
print(len(deteccoes))

# Colocando retângulos nas faces
for x, y, w, h in deteccoes:
    cv.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 5)

print(cv.imshow("color image", imagem))

# Imprimir imagem em tom de cinza
# print(cv.imshow('grayscale image',imagem_cinza))
# print(cv.imshow('color image',imagem))

# Displays image inside a window
# print(cv.imshow('color image',imagem) )
# cv.imshow('unchanged image',img_unchanged)


# Waits for a keystroke
cv.waitKey(0)

# Destroys all the windows created
# cv.destroyAllwindows()
