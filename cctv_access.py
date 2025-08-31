import cv2

ip=
port=
username=
password=
cap=cv2.VideoCapture('rtsp://')

while True:
    ret,frame=cap.read()
    if not ret:
        break

    cv2.imshow("CCTV Feed",frame)
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()