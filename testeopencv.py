import cv2
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = 'http://192.168.69.43:8080/video'
video = cv2.VideoCapture(url)

classificadorFace = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
#classificadorOlhos = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')

while True:
    conectado, frame = video.read()

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificadorFace.detectMultiScale(frameCinza, scaleFactor=1.05, minNeighbors=9, minSize=(30,30))
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
#        regiao = frame[y:y + a, x:x + l]
#        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
#        olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.02, minNeighbors=6)
#        for (ox, oy, ol, oa) in olhosDetectados:
#            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)

    cv2.imshow('Vídeo', cv2.resize(frame,(600,400)))

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()