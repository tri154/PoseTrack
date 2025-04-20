IP_ADDR = '42.114.11.251'
url1 = f'rtsp://admin:KZFADV@{IP_ADDR}:8556/ch1/main'
url2 = f'rtsp://admin:OFCTHB@{IP_ADDR}:8555/ch1/main'
footage1 = "/kaggle/input/test-video/TestVideo/output1.mp4"
footage2 = "/kaggle/input/test-video/TestVideo/output2.mp4"

cams_video = [footage1,footage2]
cams = [url1, url2]