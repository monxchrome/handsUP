import cv2
import mediapipe as mp
import pyautogui

# Инициализация библиотек
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Инициализация переменных
volume = 50  # начальная громкость
brightness = 50  # начальная яркость

# Инициализация PyAutoGUI
screen_width, screen_height = pyautogui.size()

# Функция для изменения громкости
def change_volume(direction):
    global volume
    if direction == "up":
        volume = min(volume + 5, 100)
    elif direction == "down":
        volume = max(volume - 5, 0)
    # Ваш код для изменения громкости в Windows

# Функция для изменения яркости экрана
def change_brightness(direction):
    global brightness
    if direction == "up":
        brightness = min(brightness + 5, 100)
    elif direction == "down":
        brightness = max(brightness - 5, 0)
    # Ваш код для изменения яркости экрана в Windows

# Запуск камеры
cap = cv2.VideoCapture(0)

# Запуск обработки жестов
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Обработка кадра
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Отображение результата
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ваш код для обработки жестов и изменения громкости/яркости

        # Вывод громкости и яркости на экран
        cv2.putText(frame, f"Volume: {volume}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Brightness: {brightness}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Отображение кадра
        cv2.imshow('Hand Tracking', frame)

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Очистка ресурсов
cap.release()
cv2.destroyAllWindows()
