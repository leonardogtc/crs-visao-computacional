import cv2 as cv

# Ler a imagem que será trabalhada
imagem = cv.imread("./resources/Images/clock.jpg")

# Converter a imagem para tom de cinza
imagem_cinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

# Detector carros - Carrega o xml da máquina treinada
detector_relogios = cv.CascadeClassifier("./resources/Cascades/clocks.xml")

# Roda a detecção das fases retornando em array de dados e retirando falsos positivos com "scaleFactor"
deteccoes = detector_relogios.detectMultiScale(
    imagem_cinza, scaleFactor=1.03, minNeighbors=1
)

# Colocando retângulos posicionados nas faces através do array de posições
for x, y, w, h in deteccoes:
    cv.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Carrega a imagem com os retângulos de detecção facial
print(cv.imshow("color image", imagem))

# Waits for a keystroke
cv.waitKey(0)
