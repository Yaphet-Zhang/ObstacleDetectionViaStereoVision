import cv2
import depthai as dai
import numpy as np


##### get frame
def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


##### get left mono & right mono
def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()

    # Set Camera Resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    if isLeft:
        # Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        # Get right camera
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono


##### get stereo
def getStereoPair(pipeline, monoLeft, monoRight):

    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()

    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)

    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


# ##### get mouse
# def mouseCallback(event, x, y, flags, param):
#     global mouseX, mouseY
#     if event == cv2.EVENT_LBUTTONDOWN:
#         mouseX = x
#         mouseY = y




if __name__ == '__main__':
    # mouseX = 0
    # mouseY = 640

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    # Combine left and right cameras to form a stereo pair
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    # get the output (disparity, rectifiedLeft, rectifiedRight) to connect to X-LinkOut
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")

    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    stereo.depth.link(xoutDepth.input)
    stereo.disparity.link(xoutDisp.input)
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)


    with dai.Device(pipeline) as device:
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        # maxSize, blocking: https://docs.luxonis.com/projects/api/en/latest/components/device/#specifying-arguments-for-getoutputqueue-method
        depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)

        # Calculate a multiplier for color mapping disparity map
        disparityMultiplier = 255 / stereo.getMaxDisparity()

        cv2.namedWindow("Stereo Pair")
        # cv2.setMouseCallback("Stereo Pair", mouseCallback)

        # Variable use to toggle between side by side view and one frame view.
        sideBySide = False


        c=0
        while True:
            # Get the disparity map.
            disparity = getFrame(disparityQueue) # (H:400, W:640), uint8
            # print('disparity')
            # print(disparity)

            # save
            # if c % 30 == 0:
            #     np.save('data/disparity{}'.format(str(c)), disparity) # npy


            # Get the depth map.
            depth = getFrame(depthQueue) # (400, 640), uint16
            # print('depth')
            # print(depth)

            # save
            # if c % 30 == 0:
            #     np.save('data/depth{}'.format(str(c)), depth) # npy


            # Colormap disparity for display.
            disparity = (disparity * disparityMultiplier).astype(np.uint8) # mono disparity
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET) # color disparity

            # Get the left and right rectified frame.
            leftFrame = getFrame(rectifiedLeftQueue) # (400, 640)
            rightFrame = getFrame(rectifiedRightQueue) # (400, 640)

            c+=1

            if sideBySide:
                # Show side by side view.
                imOut = np.hstack((leftFrame, rightFrame))
            else:
                # Show overlapping frames.
                imOut = np.uint8(leftFrame / 2 + rightFrame / 2)
            # Convert to RGB.
            imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)

            # # Draw scan line.
            # imOut = cv2.line(imOut, (mouseX, mouseY), (1280, mouseY), (0, 0, 255), 2)
            # # Draw clicked point.
            # imOut = cv2.circle(imOut, (mouseX, mouseY), 2, (255, 255, 128), 2)

            cv2.imshow('Stereo Pair', imOut)
            cv2.imshow('Disparity', disparity)

            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide