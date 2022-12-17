import cv2

# todo: распознавание лица openCV
if __name__ == '__main__':
    print(f'Hi')
    # загружаем классификатор
    face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # обращаемся с помощью модуля к камере
    cap = cv2.VideoCapture(0)

    while True:
        # success - результат (считалось или нет) и img - картинка
        success, img = cap.read()
        # преобразуем в оттенки серого для распознавания
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # распознаем с картинки лицо по классификатору
        faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)
        print("faces: ", faces)
        # выводим прямоугольник по координатам лица
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # выводим картинку на экран
        cv2.imshow('rez', img)
        # cv2.waitKey()
        # выходим при нажатии q
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    # освобождаем камеру
    cap.release()
    # закрываем все окна
    cv2.destroyAllWindows()
