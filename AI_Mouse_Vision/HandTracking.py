import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui


class handDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # Режим работы (по умолчанию False)
        self.maxHands = (
            maxHands  # Максимальное количество рук для отслеживания (по умолчанию 2)
        )
        self.detectionCon = (
            detectionCon  # Уровень уверенности для обнаружения руки (по умолчанию 0.5)
        )
        self.trackCon = (
            trackCon  # Уровень уверенности для отслеживания руки (по умолчанию 0.5)
        )

        self.mpHands = (
            mp.solutions.hands
        )  # Инициализация модуля Mediapipe для работы с руками
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
        )
        # Создание объекта для отслеживания рук с настройками

        self.mpDraw = (
            mp.solutions.drawing_utils
        )  # Инициализация модуля Mediapipe для рисования
        self.tipIds = [4, 8, 12, 16, 20]  # Идентификаторы кончиков пальцев

    def findHands(self, img, draw=True):
        if img is None or img.size == 0:
            print("Error: Image error.")
            return img

        imgRGB = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )  # Преобразование цветового пространства из BGR в RGB
        self.results = self.hands.process(
            imgRGB
        )  # Обработка изображения для поиска рук с использованием Mediapipe

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        # Рисование маркеров и соединительных линий рук на изображении``

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []  # Список координат x
        yList = []  # Список координат y
        bbox = []  # Границы прямоугольника вокруг обнаруженной руки
        self.lmList = []
        # Список координат ключевых точек руки

        if hasattr(self, "results") and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(
                        img, (cx, cy), 5, (0, 0, 0), cv2.FILLED
                    )  # Рисование точек ключевых точек руки

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(
                    img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2
                )  # Рисование прямоугольника вокруг руки

        return (
            self.lmList,
            bbox,
        )  # Возврат списка ключевых точек и границ прямоугольника

    def fingersUp(self):
        fingers = []  # Список, отражающий положение пальцев
        # Thumb (большой палец)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)  # Палец поднят
        else:
            fingers.append(0)  # Палец опущен

        # Fingers (остальные пальцы)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # Палец поднят
            else:
                fingers.append(0)  # Палец опущен

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(
                img, (x1, y1), (x2, y2), (0, 0, 0), t
            )  # Рисование линии между двумя точками
            cv2.circle(
                img, (x1, y1), r, (255, 0, 255), cv2.FILLED
            )  # Рисование круга вокруг первой точки
            cv2.circle(
                img, (x2, y2), r, (255, 0, 255), cv2.FILLED
            )  # Рисование круга вокруг второй точки
            cv2.circle(
                img, (cx, cy), r, (0, 0, 255), cv2.FILLED
            )  # Рисование круга вокруг центра линии
        length = math.hypot(x2 - x1, y2 - y1)  # Вычисление длины линии между точками

        return (
            length,
            img,
            [x1, y1, x2, y2, cx, cy],
        )  # Возврат длины линии, изображения с нарисованными элементами и координат центра


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = handDetector()

    while True:
        success, img = cap.read()

        if not success:
            print("Error: No read current screen.")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            # Extract coordinates of the index finger tip
            index_finger_tip = lmList[8][1:]

            # Move the mouse based on the index finger tip coordinates
            pyautogui.moveTo(index_finger_tip[0], index_finger_tip[1])

            # Check if the thumb is raised (mouse click)
            thumb_up = detector.fingersUp()[3]
            if thumb_up:
                pyautogui.click()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        if img is not None and img.size != 0:
            cv2.putText(
                img,
                str(int(fps)),
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                3,
            )
            cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
