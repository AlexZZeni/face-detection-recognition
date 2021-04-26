import cv2
import numpy as np
from PIL import Image
import os
import ssl

def cadastroRosto (stream = 0, tamanhoVideo = [640, 480], classifierDir = 'cascades\\haarcascade_frontalface_default.xml', datasetDir = 'dataset\\'):
    video = cv2.VideoCapture(stream)
    video.set(3, tamanhoVideo[0]) # largura da imagem
    video.set(4, tamanhoVideo[1]) # altura da imagem

    faceDetector = cv2.CascadeClassifier(classifierDir)
    faceID = input('\n Digite o ID do Usuario')
    faceNome = input('\n Digite o Nome do Usuario')

    i = 0
    while(True):
        ret, imagem = video.read()
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detecMultiScale(cinza, scaleFactor=1.05, minNeighbors=9, minSize=(30,30))

        for(x, y, l, a) in faces:
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            i += 1
            cv2.imwrite(datasetDir + "User." + str(faceID) + "." + str(i) + ".jpg", cinza[y:y + a, x:x + l])
            cv2.imshow("Imagem", imagem)

        if cv2.waitKey(1) == ord('q'):
            break
        elif i >= 30:
            break

    # parte que relaciona nome e id do usuario
    # necessario codificar

    video.release()
    cv2.destroyAllWindows()

def treinaRosto (classifierDir = 'cascades\\haarcascade_frontalface_default.xml', datasetDir = 'dataset\\', trainerDir = 'trainer\\trainer.yml'):
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(classifierDir)

    caminhoImagens = [os.path.join(datasetDir, f) for f in os.listdir(datasetDir)]
    amostrasFaces = []
    IDs = []

    for caminhoImagem in caminhoImagens:
        PILimg = Image.open(caminhoImagem).convert('L') # Escala de Cinza
        numpyImg = np.array(PILimg, 'uint8')
        id = int(os.path.split(caminhoImagem)[-1].split(".")[1])
        faces = detector.detectMultiScale(numpyImg)

        for(x, y, l, a) in faces:
            amostrasFaces.append(numpyImg[y:y + a, x:x + l])
            IDs.append(id)

    reconhecedor.train(amostrasFaces, np.array(IDs))
    reconhecedor.write(trainerDir)

def reconheceRosto (stream = 0, tamanhoVideo = [640, 480], classifierDir = 'cascades\\haarcascade_frontalface_default.xml', trainerDir = 'trainer\\trainer.yml'):
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read(trainerDir)
    classificador = cv2.CascadeClassifier(classifierDir)

    # parte que busca relação entre nome e id
    # necessario codificar
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    nomes = ['Vazia', 'Alexandre', 'Leonardo']

    video = cv2.VideoCapture(stream)
    video.set(3, tamanhoVideo[0]) # largura da imagem
    video.set(4, tamanhoVideo[1]) # altura da imagem
    minL = 0.1 * video.get(3)
    minA = 0.1 * video.get(4)

    while(True):
        ret, imagem = video.read()
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces = classificador.detectMultiScale(cinza, scaleFactor = 1.2, minNeighbors = 5, minSize = (int(minL), int(minA)))

        for(x, y, l, a) in faces:
            cv2.rectangle(imagem,(x, y), (x + l, y + a), (0, 0, 255), 2)
            id, confianca = reconhecedor.predict(cinza[y:y + a, x:x + l])

            if(confianca < 100):
                id = nomes[id]
                confianca = " {0}%".format(round(100 - confianca))
            else:
                id = "desconhecido"
                confianca = " {0}%".format(round(100 - confianca))

            cv2.putText(imagem, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(imagem, str(confianca), (x + 5, y + a - 5), font, 1, (255, 255, 0), 1) 
        cv2.imshow("Video",imagem)

        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
