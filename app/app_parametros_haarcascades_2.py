import cv2 as cv

# Ler a imagem que será trabalhada
imagem = cv.imread("./resources/Images/people2.jpg")

# Converter a imagem para tom de cinza
imagem_cinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

# Detector facial - Carrega o xml da máquina treinada
detector_facial = cv.CascadeClassifier(
    "./resources/Cascades/haarcascade_frontalface_default.xml"
)

# Roda a detecção das fases retornando em array de dados e retirando falsos positivos com "scaleFactor"
deteccoes = detector_facial.detectMultiScale(
    imagem_cinza, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32), maxSize=(100, 100)
)

# Colocando retângulos posicionados nas faces através do array de posições
for x, y, w, h in deteccoes:
    cv.rectangle(imagem, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Carrega a imagem com os retângulos de detecção facial
print(cv.imshow("color image", imagem))

# Waits for a keystroke
cv.waitKey(0)
