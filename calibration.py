import depthai as dai
import cv2

# pipeline
pipeline = dai.Pipeline()
# device
device = dai.Device()


# 获取相机的分辨率
print('................')
camRgb = pipeline.create(dai.node.ColorCamera) # node
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P) # set RGB resolution
resolution = camRgb.getResolutionSize() # method
print(resolution)

# # 相机标定
# calibData = device.readCalibration2()
# calibData = device.readCalibration()


# # 获取相机的内参矩阵
# # https://github.com/luxonis/depthai-experiments/blob/master/gen2-pointcloud/device-pointcloud/main.py#L147
# '''
# 3x3 intrinsics matrix

# fx   0   cx
# 0    fy  cy
# 0    0   1
# '''
# print('................')
# print(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB))
# M_right = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(resolution[0], resolution[1]),)



# # 获取相机的外参矩阵
# '''
# 4x4 intrinsics matrix
# R11  R12  R13  tx
# R21  R22  R23  ty
# R31  R32  R33  tz
# 0    0    0    1
# '''
# print('................')
# print(calibData.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))




