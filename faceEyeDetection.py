import cv2
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

#url = 'http://192.168.69.43:8080/video'
#url = 'http://192.168.137.25:8080/video'
#video = cv2.VideoCapture(url)
video = cv2.VideoCapture(0)
video.set(3,640) # set Width
video.set(4,480) # set Height

face = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
olhos = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')

while True:
    conectado, frame = video.read()

    frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetec = face.detectMultiScale(frameCinza, scaleFactor=1.05, minNeighbors=9, minSize=(30,30))

    for (x, y, l, a) in facesDetec:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = frame[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetec = olhos.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.02, minNeighbors=6)

        for (ox, oy, ol, oa) in olhosDetec:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)

    cv2.imshow('Video', cv2.resize(frame,(640,480)))

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()