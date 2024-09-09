import cv2
import time

def capture_video():
    # OpenCV VideoCapture로 /dev/video0 카메라 장치 열기
    cap = cv2.VideoCapture(0)

    # 카메라가 제대로 열렸는지 확인
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    # 해상도 및 FPS 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로 해상도 설정
    cap.set(cv2.CAP_PROP_FPS, 30)  # 프레임 속도 설정

    try:
        while True:
            # 프레임 캡처
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame.")
                break

            # 프레임을 화면에 표시
            cv2.imshow("Live Camera Feed", frame)

            # ESC 키를 눌러 종료
            if cv2.waitKey(1) & 0xFF == 27:  # ESC 키: ASCII 27
                break

    except KeyboardInterrupt:
        print("Stopping video capture...")

    finally:
        # 카메라 자원 해제
        cap.release()
        cv2.destroyAllWindows()

# 실시간 비디오 캡처 실행
capture_video()
