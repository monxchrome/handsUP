import cv2 as cv
import numpy as np
import screen_brightness_control as scb
from cvzone.HandTrackingModule import HandDetector

# Инициализация камеры
cap = cv.VideoCapture(0)

# Проверка наличия зеркального эффекта
mirror_effect = True  # Поменяйте на False, если зеркальный эффект не нужен

# Отражение изображения, если есть зеркальный эффект
if mirror_effect:
    cv.namedWindow("frame", cv.WINDOW_NORMAL)
    cv.setWindowProperty("frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# Инициализация детектора рук
hd = HandDetector()

# Настройки для управления яркостью
monitor = 0  # Уточните номер вашего дисплея
max_brightness = 100
min_brightness = 0
brightness_step = 1
current_brightness = scb.get_brightness(display=monitor)[0]

while 1:
    _, img = cap.read()

    # Отражение изображения, если есть зеркальный эффект
    if mirror_effect:
        img = cv.flip(img, 1)

    # Обнаружение рук
    hands, _ = hd.findHands(img)

    if hands and len(hands) == 1:
        # Находим координаты большого пальца для первой обнаруженной руки
        thumb = hands[0]["lmList"][4]
        thumb_x, thumb_y = thumb[0], thumb[1]

        # Проверяем, что большой палец находится справа от центра кадра
        if thumb_x < img.shape[1] // 2:
            lm = hands[0]["lmList"]
            fingers_up = all(lm[i][1] < lm[i - 2][1] for i in [8, 12, 16])

            # Изменение яркости в зависимости от положения пальцев
            if fingers_up and current_brightness < max_brightness:
                current_brightness += brightness_step
                scb.set_brightness(current_brightness, display=monitor)
            elif not fingers_up and current_brightness > min_brightness:
                current_brightness -= brightness_step
                scb.set_brightness(current_brightness, display=monitor)

    # Отображение текущей яркости на кадре
    cv.rectangle(img, (20, 150), (85, 400), (0, 255, 255), 4)
    cv.rectangle(
        img,
        (20, int(np.interp(current_brightness, [0, 100], [400, 150]))),
        (85, 400),
        (0, 0, 255),
        -1,
    )
    cv.putText(
        img,
        str(int(current_brightness)) + "%",
        (20, 430),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 0),
        3,
    )

    # Отображение кадра
    cv.imshow("frame", img)

    # Обработка клавиши для выхода из цикла
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

# Завершение работы с камерой и закрытие окна
cap.release()
cv.destroyAllWindows()
