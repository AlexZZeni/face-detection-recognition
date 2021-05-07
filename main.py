import functions
import cv2
import numpy as np
from PIL import Image
import os
import ssl

facesID = []
facesNome = []

while(True):
    opcao = input('\n Escolha uma opção:')

    if(opcao == '1'): # Cadastra novo rosto
        faceID, faceNome = functions.cadastroRosto()
        facesID.append(faceID)
        facesNome.append(faceNome)

    elif(opcao == '2'): # Traina rosto
        functions.treinaRosto()

    elif(opcao == '3'): # Reconhecimento facial
        functions.reconheceRosto()
        functions.reconheceRosto(nomes = facesNome)

    elif(opcao == '4'):
        break
