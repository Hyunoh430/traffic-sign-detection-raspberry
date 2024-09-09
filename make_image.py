import subprocess

# 저장할 이미지 경로
image_path = "image1.jpg"

# libcamera-still 명령어로 이미지 캡처
subprocess.run(["libcamera-still", "-o", image_path, "--width", "640", "--height", "480", "--timeout", "1000"])

print(f"이미지가 성공적으로 {image_path}에 저장되었습니다.")

