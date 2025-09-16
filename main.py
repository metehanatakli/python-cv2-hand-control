import cv2
import numpy as np
import time
import mediapipe as mp
import pyautogui
import win32gui
import win32con

# --- PyAutoGUI Güvenli Çıkış (Acil Durumlar İçin!) ---
# Fareyi ekranın sol üst köşesine götürdüğünde program kapanır.
pyautogui.FAILSAFE = True

# --- Ayarlanabilir Parametreler ---
CAMERA_INDEX = 0
FINGER_TIP_FILTER_WINDOW_SIZE = 3  # Fare hareketlerini yumuşatmak için filtre penceresi boyutu (daha büyük değerler daha yumuşak hareket)

# Jestler için Mesafe Eşikleri (piksel)
LEFT_HAND_CLICK_THRESHOLD = 25
RIGHT_HAND_SCROLL_THRESHOLD = 25
RIGHT_HAND_DRAG_HOLD_THRESHOLD = 30

# Aksiyon sonrası gecikme (saniye) - İstem dışı ardışık tetiklemeleri önler (Debounce süresi)
ACTION_DEBOUNCE_DELAY = 0.2

# Kaydırma Miktarı
SCROLL_AMOUNT = 20

# Ekran Boyutları
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
print(f"Ekran Çözünürlüğü: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Kamera Çözünürlüğü Ayarları (Performans ve Hassasiyet için önemli!)
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# Aktif kontrol alanı için marj (0.0 ile 1.0 arası, kameranın kenarlarından yüzde kaç içeri)
ACTIVE_ZONE_MARGIN = 0.15

# OpenCV penceresinin adı
WINDOW_NAME = "Sanal Fare Kontrolu"

# --- Göz Takibi Parametreleri (Şu an pasif) ---
EYE_TRACKING_ACTIVE = False  # Göz takibini etkinleştir/devre dışı bırak

# --- Yeni Jest Parametreleri ---
# Sağ el yumrukla geri gitme
RIGHT_HAND_FIST_BACK_THRESHOLD_TIME = 0.7  # Yumruğun ne kadar süre tutulması gerektiği (saniye)
is_right_fist_active = False  # Sağ el yumruk pozisyonunda mı?
right_fist_start_time = 0  # Yumruk pozisyonuna geçildiği zaman
has_right_fist_back_been_triggered = False  # Geri gitme eylemi tetiklendi mi?

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=2)

# Göz takibi için Face Mesh (EYE_TRACKING_ACTIVE False ise kullanılmaz)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.75,
                                  min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# --- Kamera Başlatma ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Hata: Kamera açılamadı. CAMERA_INDEX'i kontrol edin veya kamera bağlı mı?")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

ret, initial_img = cap.read()
if not ret:
    print("Hata: İlk kare okunamadı. Çıkılıyor.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

initial_img_height, initial_img_width, _ = initial_img.shape
if initial_img_width <= 0 or initial_img_height <= 0:
    print(f"Hata: Kamera boyutu geçersiz (Genişlik: {initial_img_width}, Yükseklik: {initial_img_height}).")
    print("Kamera çözünürlüğü ayarlanamadı veya kamera düzgün çalışmıyor.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# --- Jest ve Göz Takibi Durum Değişkenleri ---
mouse_position_history = []
last_mouse_x, last_mouse_y = -1, -1

is_left_click_gesture_active = False
has_left_click_been_triggered = False
last_left_click_action_time = 0

is_right_click_gesture_active = False
has_right_click_been_triggered = False
last_right_click_action_time = 0

is_scroll_down_gesture_active = False
has_scroll_down_been_triggered = False
last_scroll_down_action_time = 0

is_scroll_up_gesture_active = False
has_scroll_up_been_triggered = False
last_scroll_up_action_time = 0

is_drag_hold_gesture_active = False

# Göz Takibi İçin Değişkenler (EYE_TRACKING_ACTIVE False olduğu için pasif kalır)
gaze_position_history = []
last_gaze_x, last_gaze_y = -1, -1
dwell_start_time = 0
is_dwelling = False
last_dwell_click_time = 0
current_dwell_x, current_dwell_y = -1, -1

# --- Görsel Geri Bildirim Metinleri İçin ---
show_left_click_text = False
left_click_text_display_time = 0

show_right_click_text = False
right_click_text_display_time = 0

show_scroll_down_text = False
scroll_down_text_display_time = 0

show_scroll_up_text = False
scroll_up_text_display_time = 0

show_drag_text = False
drag_text_display_time = 0

show_gaze_text = False
gaze_text_display_time = 0
show_dwell_text = False
dwell_text_display_time = 0

show_browser_back_text = False  # Yeni: Geri gitme metni
browser_back_text_display_time = 0

print("MediaPipe Tabanlı Sanal Fare Başlatılıyor...")
print("Çıkmak için 'q' tuşuna basın veya fareyi ekranın sol üst köşesine götürün (güvenli çıkış).")
print("\n--- Jest Atamaları ---")
if EYE_TRACKING_ACTIVE:
    print("Göz Takibi AKTİF: Fare imleci göz hareketleriyle kontrol ediliyor.")
    print(f"Göz Takibi Tıklama: İmleci sabit tutarak tıklayın (varsayılan: {DWELL_TIME_THRESHOLD}s).")
else:
    print("Sol El: Fare imleci kontrolü (İşaret parmağı).")
print("Sol El (Başparmak + İşaret Parmak): Sol Tık")
print("Sol El (Başparmak + Orta Parmak): Sağ Tık")
print("Sağ El (Başparmak + İşaret Parmak): Aşağı Kaydırma")
print("Sağ El (Başparmak + Orta Parmak): Yukarı Kaydırma")
print("Sağ El (Başparmak + Yüzük Parmak): Sol Tık Basılı Tutma (Sürükle-Bırak)")
print(f"Sağ El (Yumruk Yapıp Tutma): GERİ GİT (varsayılan: {RIGHT_HAND_FIST_BACK_THRESHOLD_TIME}s)")

prev_frame_time = 0
new_frame_time = 0

# --- Pencereyi Her Zaman En Önde Tutma Ayarları ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
hwnd = win32gui.FindWindow(None, WINDOW_NAME)

if hwnd:
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    print(f"'{WINDOW_NAME}' penceresi her zaman en önde tutuluyor.")
else:
    print(f"Uyarı: '{WINDOW_NAME}' penceresi bulunamadı. 'Her zaman en önde' özelliği aktif olmayabilir.")


# --- Yardımcı Fonksiyon: Parmakların Kapalı Olup Olmadığını Kontrol Etme ---
# El landmark'ları verilere göre bir parmağın kapalı olup olmadığını kontrol eder.
# Basitçe, parmak ucunun parmağın tabanına olan Y koordinatına göre yukarıda olup olmadığına bakar.
# Düşük parmak ucunun Y koordinatı, yüksek parmak tabanının Y koordinatından büyükse kapalıdır (aşağıda ise açık).
def is_finger_closed(landmarks_list, tip_idx, pip_idx, mcp_idx, direction='vertical'):
    # landmarks_list artık doğrudan hand_landmarks objesi, içindeki landmark'lara .landmark ile erişiyoruz
    tip_y = landmarks_list.landmark[tip_idx].y
    pip_y = landmarks_list.landmark[pip_idx].y
    mcp_y = landmarks_list.landmark[mcp_idx].y

    tip_x = landmarks_list.landmark[tip_idx].x
    pip_x = landmarks_list.landmark[pip_idx].x
    mcp_x = landmarks_list.landmark[mcp_idx].x

    if direction == 'vertical':  # Dikey kapanma (yumruk için)
        # Parmak ucu parmak kökünden daha aşağıda (kamerada y değeri arttıkça aşağıya iner)
        # Yumruk yapıldığında parmak uçları genelde pip ve mcp'den daha aşağıda kalır.
        return tip_y > pip_y and pip_y > mcp_y
    elif direction == 'horizontal':  # Yatay kapanma (başparmak için, nadiren kullanılır)
        # Parmak ucunun ve diğer boğumların X mesafesi çok azsa
        return abs(tip_x - pip_x) < 0.05 and abs(pip_x - mcp_x) < 0.05
    return False


# Tüm parmakların kapalı olup olmadığını kontrol et
def is_hand_fist(hand_landmarks):
    # hand_landmarks doğrudan MediaPipe'in verdiği NormalizedLandmarkList objesi
    # Her bir landmark'a .landmark[index] ile erişiyoruz.

    # Başparmak (thumb) için özel kontrol: ucu ve dibi arasındaki mesafe küçükse
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    # thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP] # İç boğum
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # Kök

    # Başparmak ucu, köke göre X ekseninde yakın ve Y ekseninde genellikle daha aşağıda olmalı
    # Bu kontrol, yumruk yapıldığında başparmağın da avuç içine doğru katlanmasını hedefler.
    # Geniş bir eşik verelim ki farklı el yapılarına uyum sağlasın.
    dist_thumb_mcp_tip = np.sqrt((thumb_tip.x - thumb_mcp.x) ** 2 + (thumb_tip.y - thumb_mcp.y) ** 2)
    # Yumruk yapıldığında başparmak ucu, başparmağın metakarposuna (mcp) çok yakın olmalı.
    # Normalize edilmiş koordinatlar 0-1 arası olduğu için 0.1 gibi bir eşik uygun.
    is_thumb_folded = dist_thumb_mcp_tip < 0.1

    # Diğer dört parmak için (işaret, orta, yüzük, serçe)
    is_index_closed = is_finger_closed(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                       mp_hands.HandLandmark.INDEX_FINGER_PIP,
                                       mp_hands.HandLandmark.INDEX_FINGER_MCP)
    is_middle_closed = is_finger_closed(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                                        mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    is_ring_closed = is_finger_closed(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP,
                                      mp_hands.HandLandmark.RING_FINGER_PIP,
                                      mp_hands.HandLandmark.RING_FINGER_MCP)
    is_pinky_closed = is_finger_closed(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP,
                                       mp_hands.HandLandmark.PINKY_PIP,
                                       mp_hands.HandLandmark.PINKY_MCP)

    # Tüm parmaklar (başparmak dahil) kapalıysa yumruktur.
    return is_thumb_folded and is_index_closed and is_middle_closed and is_ring_closed and is_pinky_closed


# --- Ana Program Döngüsü ---
while True:
    success, img = cap.read()
    if not success:
        print("Kamera okunamadı. Çıkılıyor.")
        break

    img_height, img_width, _ = img.shape
    if img_width <= 0 or img_height <= 0:
        print("Uyarı: Geçersiz görüntü boyutu (0x0). Atlanıyor.")
        continue

    img = cv2.flip(img, 1)  # Yatay çevir

    # Pencereyi her zaman en önde tut
    if hwnd:
        current_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        if not (current_style & win32con.WS_EX_TOPMOST):
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    else:
        hwnd = win32gui.FindWindow(None, WINDOW_NAME)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    # --- FPS Hesaplama ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(img, fps_text, (img_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Göz Takibi (EYE_TRACKING_ACTIVE False olduğu için pasif kalır) ---
    gaze_detected = False
    if EYE_TRACKING_ACTIVE:
        face_results = face_mesh.process(img_rgb)
        # Göz takibi kodu buraya gelmeliydi, ama şu an pasif.
        # Devre dışı bırakıldığı için bu kısım atlanacak.
        pass  # Geçici olarak boş bırakıldı

    # --- El Algılama ve Jestler ---
    hand_results = hands.process(img_rgb)
    left_hand_active = False
    right_hand_active = False

    temp_is_left_click_gesture_active = False
    temp_is_right_click_gesture_active = False
    temp_is_scroll_down_gesture_active = False
    temp_is_scroll_up_gesture_active = False
    temp_is_drag_hold_gesture_active = False
    temp_is_right_fist_active = False  # Yumruk durumu için geçici değişken

    if hand_results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            handedness = hand_results.multi_handedness[hand_idx].classification[0].label

            # Sol el algılandı
            if handedness == "Left":
                left_hand_active = True
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Fare İmleci Hareketi (Göz Takibi aktifse bu kısım çalışmaz)
                if not EYE_TRACKING_ACTIVE:
                    src_x_min = ACTIVE_ZONE_MARGIN
                    src_x_max = 1.0 - ACTIVE_ZONE_MARGIN
                    src_y_min = ACTIVE_ZONE_MARGIN
                    src_y_max = 1.0 - ACTIVE_ZONE_MARGIN

                    dest_x_min = 0
                    dest_x_max = SCREEN_WIDTH
                    dest_y_min = 0
                    dest_y_max = SCREEN_HEIGHT

                    mapped_x = np.interp(index_finger_tip.x, [src_x_min, src_x_max], [dest_x_min, dest_x_max])
                    mapped_y = np.interp(index_finger_tip.y, [src_y_min, src_y_max], [dest_y_min, dest_y_max])

                    mapped_x = np.clip(mapped_x, 0, SCREEN_WIDTH - 1)
                    mapped_y = np.clip(mapped_y, 0, SCREEN_HEIGHT - 1)

                    mouse_position_history.append((mapped_x, mapped_y))
                    if len(mouse_position_history) > FINGER_TIP_FILTER_WINDOW_SIZE:
                        mouse_position_history.pop(0)

                    avg_mouse_x = sum([p[0] for p in mouse_position_history]) // len(mouse_position_history)
                    avg_mouse_y = sum([p[1] for p in mouse_position_history]) // len(mouse_position_history)

                    current_mouse_x, current_mouse_y = int(avg_mouse_x), int(avg_mouse_y)

                    if last_mouse_x != current_mouse_x or last_mouse_y != current_mouse_y:
                        pyautogui.moveTo(current_mouse_x, current_mouse_y, _pause=False)
                        last_mouse_x, last_mouse_y = current_mouse_x, current_mouse_y

                    cam_x_index_left = int(index_finger_tip.x * img_width)
                    cam_y_index_left = int(index_finger_tip.y * img_height)
                    cv2.circle(img, (cam_x_index_left, cam_y_index_left), 15, (255, 0, 255), cv2.FILLED)
                    cv2.putText(img, "Sol El (Fare)", (cam_x_index_left + 20, cam_y_index_left + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # Sol El ile tıklama jestleri (Göz takibi aktif olsa da çalışır)
                thumb_x = int(thumb_tip.x * img_width)
                thumb_y = int(thumb_tip.y * img_height)

                # Sol Tıklama (Başparmak + İşaret Parmak)
                cam_x_index_left = int(index_finger_tip.x * img_width)
                cam_y_index_left = int(index_finger_tip.y * img_height)
                dist_thumb_index_left = np.sqrt((cam_x_index_left - thumb_x) ** 2 + (cam_y_index_left - thumb_y) ** 2)
                cv2.line(img, (cam_x_index_left, cam_y_index_left), (thumb_x, thumb_y), (0, 255, 0), 2)
                cv2.putText(img, f"LClick Mesafe: {int(dist_thumb_index_left)}",
                            (cam_x_index_left + 20, cam_y_index_left - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                if dist_thumb_index_left < LEFT_HAND_CLICK_THRESHOLD:
                    temp_is_left_click_gesture_active = True
                    if not has_left_click_been_triggered and (
                            time.time() - last_left_click_action_time > ACTION_DEBOUNCE_DELAY):
                        pyautogui.click(button='left')
                        print("Sol El: Sol Tıklandı!")
                        has_left_click_been_triggered = True
                        last_left_click_action_time = time.time()
                        show_left_click_text = True
                        left_click_text_display_time = time.time()
                else:
                    has_left_click_been_triggered = False

                # Sağ Tıklama (Başparmak + Orta Parmak)
                cam_x_middle_left = int(middle_finger_tip.x * img_width)
                cam_y_middle_left = int(middle_finger_tip.y * img_height)
                dist_thumb_middle_left = np.sqrt(
                    (cam_x_middle_left - thumb_x) ** 2 + (cam_y_middle_left - thumb_y) ** 2)
                cv2.line(img, (cam_x_middle_left, cam_y_middle_left), (thumb_x, thumb_y), (0, 165, 255), 2)
                cv2.putText(img, f"RClick Mesafe: {int(dist_thumb_middle_left)}",
                            (cam_x_middle_left + 20, cam_y_middle_left - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                if dist_thumb_middle_left < LEFT_HAND_CLICK_THRESHOLD:
                    temp_is_right_click_gesture_active = True
                    if not has_right_click_been_triggered and (
                            time.time() - last_right_click_action_time > ACTION_DEBOUNCE_DELAY):
                        pyautogui.click(button='right')
                        print("Sol El: Sağ Tıklandı!")
                        has_right_click_been_triggered = True
                        last_right_click_action_time = time.time()
                        show_right_click_text = True
                        right_click_text_display_time = time.time()
                else:
                    has_right_click_been_triggered = False

            # Sağ el algılandı
            elif handedness == "Right":
                right_hand_active = True
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                thumb_x = int(thumb_tip.x * img_width)
                thumb_y = int(thumb_tip.y * img_height)

                # Sağ el yumruk kontrolü (Yeni Jest)
                if is_hand_fist(hand_landmarks):
                    temp_is_right_fist_active = True
                    if not is_right_fist_active:  # Yumruk yeni yapıldı
                        right_fist_start_time = time.time()
                        is_right_fist_active = True
                        print("Sağ el yumruk yapıldı, geri gitme için bekleniyor...")

                    # Yeterli süre yumruk tutulduysa ve henüz tetiklenmediyse
                    if (time.time() - right_fist_start_time >= RIGHT_HAND_FIST_BACK_THRESHOLD_TIME) and \
                            (not has_right_fist_back_been_triggered) and \
                            (time.time() - last_left_click_action_time > ACTION_DEBOUNCE_DELAY) and \
                            (time.time() - last_right_click_action_time > ACTION_DEBOUNCE_DELAY) and \
                            (time.time() - last_scroll_down_action_time > ACTION_DEBOUNCE_DELAY) and \
                            (time.time() - last_scroll_up_action_time > ACTION_DEBOUNCE_DELAY):
                        pyautogui.press('browserback')  # Geri gitme komutu
                        print(f"Sağ El: Yumruk tutarak GERİ GİT yapıldı!")
                        has_right_fist_back_been_triggered = True
                        show_browser_back_text = True
                        browser_back_text_display_time = time.time()
                        # Yumruk bırakılana kadar bir daha tetiklememek için
                        # right_fist_start_time = 0 # Sıfırlamayabiliriz, zaten flag var
                else:  # Yumruk pozisyonunda değilse sıfırla
                    is_right_fist_active = False
                    right_fist_start_time = 0
                    has_right_fist_back_been_triggered = False

                # Aşağı Kaydırma (Başparmak + İşaret Parmak)
                cam_x_index_right = int(index_finger_tip.x * img_width)
                cam_y_index_right = int(index_finger_tip.y * img_height)
                dist_thumb_index_right = np.sqrt(
                    (cam_x_index_right - thumb_x) ** 2 + (cam_y_index_right - thumb_y) ** 2)
                cv2.line(img, (cam_x_index_right, cam_y_index_right), (thumb_x, thumb_y), (0, 100, 255), 2)
                cv2.putText(img, f"Scroll Down: {int(dist_thumb_index_right)}",
                            (cam_x_index_right + 20, cam_y_index_right - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                if dist_thumb_index_right < RIGHT_HAND_SCROLL_THRESHOLD:
                    temp_is_scroll_down_gesture_active = True
                    if not has_scroll_down_been_triggered and (
                            time.time() - last_scroll_down_action_time > ACTION_DEBOUNCE_DELAY):
                        pyautogui.scroll(-SCROLL_AMOUNT)
                        print(f"Sağ El: {SCROLL_AMOUNT} birim aşağı kaydırıldı!")
                        has_scroll_down_been_triggered = True
                        last_scroll_down_action_time = time.time()
                        show_scroll_down_text = True
                        scroll_down_text_display_time = time.time()
                else:
                    has_scroll_down_been_triggered = False

                # Yukarı Kaydırma (Başparmak + Orta Parmak)
                cam_x_middle_right = int(middle_finger_tip.x * img_width)
                cam_y_middle_right = int(middle_finger_tip.y * img_height)
                dist_thumb_middle_right = np.sqrt(
                    (cam_x_middle_right - thumb_x) ** 2 + (cam_y_middle_right - thumb_y) ** 2)
                cv2.line(img, (cam_x_middle_right, cam_y_middle_right), (thumb_x, thumb_y), (0, 200, 0), 2)
                cv2.putText(img, f"Scroll Up: {int(dist_thumb_middle_right)}",
                            (cam_x_middle_right + 20, cam_y_middle_right - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                if dist_thumb_middle_right < RIGHT_HAND_SCROLL_THRESHOLD:
                    temp_is_scroll_up_gesture_active = True
                    if not has_scroll_up_been_triggered and (
                            time.time() - last_scroll_up_action_time > ACTION_DEBOUNCE_DELAY):
                        pyautogui.scroll(SCROLL_AMOUNT)
                        print(f"Sağ El: {SCROLL_AMOUNT} birim yukarı kaydırıldı!")
                        has_scroll_up_been_triggered = True
                        last_scroll_up_action_time = time.time()
                        show_scroll_up_text = True
                        scroll_up_text_display_time = time.time()
                else:
                    has_scroll_up_been_triggered = False

                # Sol Tık Basılı Tutma / Sürükle-Bırak (Başparmak + Yüzük Parmak)
                cam_x_ring_right = int(ring_finger_tip.x * img_width)
                cam_y_ring_right = int(ring_finger_tip.y * img_height)
                dist_thumb_ring_right = np.sqrt((cam_x_ring_right - thumb_x) ** 2 + (cam_y_ring_right - thumb_y) ** 2)
                cv2.line(img, (cam_x_ring_right, cam_y_ring_right), (thumb_x, thumb_y), (255, 0, 0), 2)
                cv2.putText(img, f"Drag Hold: {int(dist_thumb_ring_right)}",
                            (cam_x_ring_right + 20, cam_y_ring_right + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)

                if dist_thumb_ring_right < RIGHT_HAND_DRAG_HOLD_THRESHOLD:
                    temp_is_drag_hold_gesture_active = True
                    if not is_drag_hold_gesture_active:
                        pyautogui.mouseDown(button='left', _pause=False)
                        print("Sağ El: Sol Tık Basılı Tutuluyor (Sürükleme Başladı!)")
                        show_drag_text = True
                        drag_text_display_time = time.time()
                else:
                    if is_drag_hold_gesture_active:
                        pyautogui.mouseUp(button='left', _pause=False)
                        print("Sağ El: Sol Tık Bırakıldı (Sürükleme Bitti!)")
                        show_drag_text = False
                    temp_is_drag_hold_gesture_active = False

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- El Algılama Sonrası Jest Durumlarını Güncelle ---
    if not temp_is_left_click_gesture_active:
        has_left_click_been_triggered = False
    if not temp_is_right_click_gesture_active:
        has_right_click_been_triggered = False

    if not temp_is_scroll_down_gesture_active:
        has_scroll_down_been_triggered = False
    if not temp_is_scroll_up_gesture_active:
        has_scroll_up_been_triggered = False

    if is_drag_hold_gesture_active and not temp_is_drag_hold_gesture_active:
        pyautogui.mouseUp(button='left', _pause=False)
        print("Sürükleme: Sağ El Kaybolduğu/Ayrıldığı İçin Sol Tık Bırakıldı!")
        show_drag_text = False
    is_drag_hold_gesture_active = temp_is_drag_hold_gesture_active

    # --- El Algılanmadığında İlgili Jest Durumlarını Sıfırla ---
    if not left_hand_active:
        mouse_position_history.clear()
        last_mouse_x, last_mouse_y = -1, -1
        has_left_click_been_triggered = False
        has_right_click_been_triggered = False

    if not right_hand_active:
        has_scroll_down_been_triggered = False
        has_scroll_up_been_triggered = False
        if is_drag_hold_gesture_active:
            pyautogui.mouseUp(button='left', _pause=False)
            print("Sürükleme: Sağ El Kaybolduğu İçin Sol Tık Bırakıldı!")
            show_drag_text = False
        is_drag_hold_gesture_active = False

        # Sağ el kaybolduğunda veya yumruk pozisyonunda değilken yumruk durumunu sıfırla
        is_right_fist_active = False
        right_fist_start_time = 0
        has_right_fist_back_been_triggered = False

    # --- Göz Algılanmadığında Göz Takibi Durumlarını Sıfırla (Pasif) ---
    if EYE_TRACKING_ACTIVE and not gaze_detected:
        # Göz takibi kodu aktif olsaydı burada sıfırlama yapılırdı
        pass  # Geçici olarak boş bırakıldı

    # --- Görsel Geri Bildirim Metinlerini Yönet ---
    current_time = time.time()
    text_y_offset = 0

    if show_left_click_text and (current_time - left_click_text_display_time < 0.7):
        cv2.putText(img, "SOL TIKLANDI!", (50, img_height - 100 + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2)
    elif show_left_click_text and (current_time - left_click_text_display_time >= 0.7):
        show_left_click_text = False

    if show_right_click_text and (current_time - right_click_text_display_time < 0.7):
        cv2.putText(img, "SAG TIKLANDI!", (50, img_height - 70 + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 0, 0), 2)
    elif show_right_click_text and (current_time - right_click_text_display_time >= 0.7):
        show_right_click_text = False

    right_text_x_start = img_width - 350
    if show_scroll_down_text and (current_time - scroll_down_text_display_time < 0.7):
        cv2.putText(img, "ASAGI KAYDIRILIYOR!", (right_text_x_start, img_height - 100 + text_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
    elif show_scroll_down_text and (current_time - scroll_down_text_display_time >= 0.7):
        show_scroll_down_text = False

    if show_scroll_up_text and (current_time - scroll_up_text_display_time < 0.7):
        cv2.putText(img, "YUKARI KAYDIRILIYOR!", (right_text_x_start, img_height - 70 + text_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
    elif show_scroll_up_text and (current_time - scroll_up_text_display_time >= 0.7):
        show_scroll_up_text = False

    if show_drag_text:
        cv2.putText(img, "SURUKLENIYOR!", (right_text_x_start, img_height - 40 + text_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    if show_browser_back_text and (current_time - browser_back_text_display_time < 0.7):  # Yeni: Geri gitme metni
        cv2.putText(img, "GERI GITTI!", (right_text_x_start, img_height - 10 + text_y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 100, 0), 2)
    elif show_browser_back_text and (current_time - browser_back_text_display_time >= 0.7):
        show_browser_back_text = False

    # Göz Takibi Metinleri (EYE_TRACKING_ACTIVE False olduğu için pasif kalır)
    if EYE_TRACKING_ACTIVE:
        # Göz takibi metinleri burada gösterilecekti, ama şu an pasif.
        pass  # Geçici olarak boş bırakıldı

    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
