#!/usr/bin/env python3
import cv2
import numpy as np
import depthai as dai
import time
import numpy as np
import cv2
from utils.point_cloud_process import viz_o3d_pc, np_pc_to_o3d_pc, o3d_pc_to_np_pc, calc_outlier, depth_to_pc
from utils.point_cloud_process import viz_inlier_outlier, calc_array2d_diff, calc_array2d_diff_index, cal_color, np_to_o3d
from utils.point_cloud_process import box_intersection_volume
import matplotlib.pyplot as plt
import open3d as o3d
import time
import numpy as np
# import k3d
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from scipy.signal import find_peaks
from utils.obstacle_detection import calc1




# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True
# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6
# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p: (720, 1280, 3)
# Otherwise (False), the aligned depth is automatically upscaled to 1080p: (1080, 1920, 3)
downscaleColor = True
fps = 30
threshold = 180 # 0~255 
# sidebyside or one-screen show the left & right.
sideBySide = False


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image
    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb) / 100.0
    depthWeight = 1.0 - rgbWeight




#################### Create pipeline ####################
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources
camRgb = pipeline.create(dai.node.ColorCamera) # RGB
left = pipeline.create(dai.node.MonoCamera) # left
right = pipeline.create(dai.node.MonoCamera) # right
stereo = pipeline.create(dai.node.StereoDepth) # disparity

# Define outputs
rgbOut = pipeline.create(dai.node.XLinkOut)
disparityOut = pipeline.create(dai.node.XLinkOut)
leftOut = pipeline.create(dai.node.XLinkOut)
rightOut = pipeline.create(dai.node.XLinkOut)

# Add queue
rgbOut.setStreamName("rgb")
queueNames.append("rgb")
leftOut.setStreamName("rectifiedLeft")
queueNames.append("rectifiedLeft")
rightOut.setStreamName("rectifiedRight")
queueNames.append("rectifiedRight")
disparityOut.setStreamName("disp")
queueNames.append("disp")




#################### Properties ####################

##### Stereo
# Fixing noisy depth -> Improving depth accuracy -> Long range stereo depth
# https://docs.luxonis.com/projects/api/en/latest/tutorials/configuring-stereo-depth/?highlight=pointcloud#stereo-depth-confidence-threshold
# stereo.initialConfig.setConfidenceThreshold(threshold) # Or, alternatively, set the Stereo Preset Mode below:
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY) # Prioritize fill-rate, sets Confidence threshold to 245
# stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY) # Prioritize accuracy, sets Confidence threshold to 200

# The disparity is computed at this resolution, then upscaled to RGB resolution
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P) # set left/depth resolution
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
left.setFps(fps)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P) # set right/depth resolution
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
right.setFps(fps)

# post-processing filter for depth smoothness
# https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/depth_post_processing/#depth-post-processing
# median filter
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7) # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)

stereo.setExtendedDisparity(extended_disparity)

# https://docs.luxonis.com/projects/api/en/latest/tutorials/configuring-stereo-depth/?highlight=pointcloud#stereo-subpixel-mode
stereo.setSubpixel(subpixel)
# brightness filter (invalidate light/corner noise. default: 0, or int8)
stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
# Host-side pointcloud filter (need Open3D or PCL)

config = stereo.initialConfig.get()
# speckle filter
config.postProcessing.speckleFilter.enable = False # True?
config.postProcessing.speckleFilter.speckleRange = 50
# # temporal filter (slow) (for static)
# config.postProcessing.temporalFilter.enable = True
# # spatial filter (slow)
# config.postProcessing.spatialFilter.enable = True
# config.postProcessing.spatialFilter.holeFillingRadius = 2
# config.postProcessing.spatialFilter.numIterations = 1
# threshold filter
config.postProcessing.thresholdFilter.minRange = 400 # mm
config.postProcessing.thresholdFilter.maxRange = 15000 # mm
# # decimation filter
# config.postProcessing.decimationFilter.decimationFactor = 1
# # ??
# config.censusTransform.enableMeanMode = True
# config.costMatching.linearEquationParameters.alpha = 0
# config.costMatching.linearEquationParameters.beta = 2
stereo.initialConfig.set(config)

# LR-check is required for depth alignment
stereo.setLeftRightCheck(lr_check)
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
# stereo.setOutputSize(640, 400) ### for 400P (error)

##### RGB
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # set RGB resolution
camRgb.setFps(fps)

if downscaleColor: camRgb.setIspScale(2, 3) # 1st param : 2nd param = rgb/stereo : 1080p

# For now, RGB needs fixed focus to properly align with depth.
# This value was used during calibration
try:
    calibData = device.readCalibration2() # calibration parameters
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)
except:
    raise




#################### Linking ####################
camRgb.isp.link(rgbOut.input)
stereo.rectifiedLeft.link(leftOut.input)
stereo.rectifiedRight.link(rightOut.input)
left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.disparity.link(disparityOut.input)




#################### Connect to device and start pipeline ####################
with device:
    device.startPipeline(pipeline)
    frameRgb = None
    leftFrame = None
    rightFrame = None
    frameDisp = None

    # Configure windows; trackbar adjusts blending ratio of rgb/depth
    # rgbWindowName = "rgb"
    # lrWindowName = "left-right"
    # depthWindowName = "disparity"
    blendedWindowName = "rgb-disparity"
    # cv2.namedWindow(rgbWindowName)
    # cv2.namedWindow(lrWindowName)
    # cv2.namedWindow(depthWindowName)
    cv2.namedWindow(blendedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)


    #################### write RGB video ####################
    # size = (1280, 720)
    # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # writer = cv2.VideoWriter('./result/video/out.mp4', fmt, fps, size)

    c = 0
    while True:
        start_time = time.time() # start time

        latestPacket = {}
        latestPacket["rgb"] = None
        latestPacket["rectifiedLeft"] = None
        latestPacket["rectifiedRight"] = None
        latestPacket["disp"] = None

        queueEvents = device.getQueueEvents(("rgb", "rectifiedLeft", "rectifiedRight", "disp"))
        for queueName in queueEvents:
            packets = device.getOutputQueue(queueName).tryGetAll()
            if len(packets) > 0:
                latestPacket[queueName] = packets[-1]


        #################### RGB ####################
        if latestPacket["rgb"] is not None:
            frameRgb = latestPacket["rgb"].getCvFrame()
            # cv2.imshow(rgbWindowName, frameRgb) # RGB: (720, 1280, 3)

            # save
            print('rgb', frameRgb.shape)
            # np.save('../data/0000/rgb{}'.format(str(c)), frameRgb) # npy

            # writer.write(frameRgb)


        #################### left & right ####################
        if latestPacket["rectifiedLeft"] is not None and latestPacket["rectifiedRight"] is not None:
            leftFrame = latestPacket["rectifiedLeft"].getCvFrame() # left: (720, 1280), [0, 255]
            rightFrame = latestPacket["rectifiedRight"].getCvFrame()
            if sideBySide:
                # Show side by side view.
                imOut = np.hstack((leftFrame, rightFrame))
            else:
                # Show overlapping frames.
                imOut = np.uint8(leftFrame / 2 + rightFrame / 2)

            # save
            print('lr', imOut.shape) 
            # np.save('../data/0000/leftright{}'.format(str(c)), imOut) # npy   

            # Convert to RGB.
            imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB) # only copy 1 channel to 3 channels  
            # imOut_resized = cv2.resize(imOut, (640, 400)) 
            # cv2.imshow(lrWindowName, imOut)


        #################### disparity ####################
        if latestPacket["disp"] is not None:
            frameDisp = latestPacket["disp"].getFrame() # original disparity: (720, 1280), [0~95]

            # save
            print('disp', frameDisp.shape) 
            # np.save('../data/0000/disparity{}'.format(str(c)), frameDisp) # npy            


            ################### Depth calculatin from disparity ####################
            # https://docs.luxonis.com/projects/api/en/latest/tutorials/configuring-stereo-depth/?highlight=focal%20length#stereo-depth-basics
            # depth[mm] = focalLength[pix] ∗ baseline[mm] / disparity[pix] # oak-d baseline: 75mm
            # focalLength[pix] = widthpx[pix] ∗ 0.5 / tan(hfov[deg] ∗ 0.5 ∗ pi/180) # oak-d stereo HFOV: 71.9 degrees
            # if 400P(640x400) , widthpx:640, focalLength:441.25
            # if 800P(1280x800) , widthpx:1280, focalLength:882.5
            focalLength = 1280 * 0.5 / np.tan(71.9 * 0.5 * np.pi / 180)
            # only calc nonzero
            frameDepth = np.where(frameDisp!=0.0, (focalLength*75/(frameDisp)).astype(np.uint16), frameDisp) # depth map: (720, 1280)

            # save
            print('depth', frameDepth.shape) 
            # np.save('../data/0000/depth{}'.format(str(c)), frameDepth) # npy


            # calc1(frameRgb, frameDepth)


            ################### only for viz ####################
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDisp = (frameDisp * (255/stereo.initialConfig.getMaxDisparity())).astype(np.uint8) # mono disparity (0~255)
            # Optional, apply false colorization
            if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET) # color disparity (cv2.COLORMAP_JET)
            frameDisp = np.ascontiguousarray(frameDisp)
            # cv2.imshow(depthWindowName, frameDisp) # Disparity: (720, 1280, 3)


        ################### disparity + RGB ####################
        if frameRgb is not None and frameDisp is not None:
            # Need to have both frames in BGR format before blending
            if len(frameDisp.shape) < 3: # only when disparity is 1 grayscale
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
            blended_resized = cv2.resize(blended, (640, 400)) # just for show
            cv2.imshow(blendedWindowName, blended_resized) # Disparity + RGB (720, 1280, 3)

            frameRgb = None
            frameDisp = None


        ################### time ###################
        # time.sleep(0.01) # stereo wait for rgb
        end_time = time.time() # end time
        if end_time-start_time != 0:
            print('----------fps:{:.2f}----------'.format(1/(end_time - start_time)))
        c += 1


        ################### end ###################
        key = cv2.waitKey(1)
        if key == ord('q'): # Quit when q is pressed
            break
        elif key == ord('t'): # Toggle display when t is pressed
            sideBySide = not sideBySide



    # writer.release()
    # cv2.destroyAllWindows()
            
