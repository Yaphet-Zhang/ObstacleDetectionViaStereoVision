import cv2

cap = cv2.VideoCapture(r'./result/video/out.mp4')

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 1280
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 720
num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 53259
fps = int(cap.get(cv2.CAP_PROP_FPS)) # 30

c = 0
for i in range(num_frame):
    ret, frame = cap.read()

    if i % (4*fps) == 0:
        cv2.imwrite('./result/image/{0:04}.jpg'.format(c), frame) # save image as 0000 -> 9999
        print('saved:', './result/image/{0:04}.jpg'.format(c))
        c += 1

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




