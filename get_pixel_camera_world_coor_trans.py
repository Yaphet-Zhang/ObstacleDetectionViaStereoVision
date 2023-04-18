'''
# https://dev.classmethod.jp/articles/convert-coords-screen-to-space/
# https://dev.classmethod.jp/articles/convert-coordinates/


import numpy as np
from numpy import sin, cos, tan


##########
def convert_uvz_to_xyz(u, v, z, R, t, K):
    K_inv = np.linalg.inv(K)

    # in screen coord
    cs = np.asarray([u, v, 1])
    cs_ = cs * z

    # in camera coord
    cc = np.dot(K_inv, cs_)

    # in world coord
    cw = np.dot(R, cc) + t

    return cw

    
##########
R_unreal_to_world = np.asarray([
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0],
])

def convert_coord_unreal_to_world(cu):
    # coord in world
    cw = np.dot(R_unreal_to_world, cu)
    return cw

def convert_coord_world_to_unreal(cw):
    R_unreal_to_world_inv = np.linalg.inv(R_unreal_to_world)

    # coord in unrealengine
    cu = np.dot(R_unreal_to_world_inv, cw)
    return cu


    

# UnrealEngine座標系におけるカメラの座標
cam_coord = [392, 336, 234]
# カメラの回転角度 (カメラ座標におけるx軸、y軸、z軸での回転)
cam_rot = [326, 41, 0]
# カメラの視野角（水平方向）
fov = 90
# スクリーンの画素数（横）
pw = 1280
# スクリーンの画素数（縦）
ph = 720
# カメラ情報（内部パラメータ）
cam_info = (fov, pw, ph)

# 対応する座標の組み合わせ
# uvz(スクリーン座標+Depth), xyz(UnrealEngine座標) の順
coord_set = [
    ([663, 306, 263], [560, 500, 100]),
    ([776, 366, 227], [500, 500, 100]),
    ([935, 453, 189], [440, 500, 100]),
    ([758, 263, 297], [560, 560, 100]),
    ([871, 313, 258], [500, 560, 100]),
    ([1022, 376, 222], [440, 560, 100]),
    ([661, 402, 291], [560, 500, 50]),
    ([762, 469, 255], [500, 500, 50]),
    ([898, 562, 218], [440, 500, 50]),
    ([748, 354, 324], [560, 560, 50]),
    ([849, 408, 287], [500, 560, 50]),
    ([978, 481, 250], [440, 560, 50]),
]




##########
def calc_R(pitch, yaw, roll):
    a = np.radians(pitch)
    b = np.radians(yaw)
    c = np.radians(roll)

    R_x = np.asarray([
        [1, 0, 0],
        [0, cos(a), -sin(a)],
        [0, sin(a), cos(a)],
    ])

    R_y = np.asarray([
        [cos(b), 0, sin(b)],
        [0, 1, 0],
        [-sin(b), 0, cos(b)],
    ])

    R_z = np.asarray([
        [cos(c), -sin(c), 0],
        [sin(c), cos(c), 0],
        [0, 0, 1],
    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

    

def calc_K(fov_x, pixel_w, pixel_h, cx=None, cy=None):
    if cx is None:
        cx = pixel_w / 2.0
    if cy is None:
        cy = pixel_h / 2.0

    fx = 1.0 / (2.0 * tan(np.radians(fov_x) / 2.0)) * pixel_w
    fy = fx

    K = np.asarray([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    return K

def calc_t(camera_coord):
    return convert_coord_unreal_to_world(camera_coord)

t = calc_t(cam_coord)
R = calc_R(*cam_rot)
K = calc_K(*cam_info)


for cs, cu in coord_set:
    u, v, z = cs
    # 変換によって推定したワールド座標系での座標
    cw_ = convert_uvz_to_xyz(u, v, z, R, t, K)
    # ワールド座標をUnrealEngine座標に変換する
    cu_ = convert_coord_world_to_unreal(cw_)

    print(cs)
    print(cu_)
    print(cu)
    print()
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab
import open3d as o3d


# https://blog.csdn.net/a40850273/article/details/122410448
 
# 加载深度数据
depth = np.load('./data/depth100.npy') # (H:400, W:640)
img = depth.astype(np.float32)

# 参数
CAM_WID, CAM_HGT = 640, 400
CAM_FX, CAM_FY = 795.209, 793.957 # fx，fy 分别为镜头 x，y 方向的焦距（成像平面到镜头的距离）
CAM_CX, CAM_CY = 332.031, 231.308 # cx，cy 分别是光心（镜头中心点在成像平面的投影）在图像坐标系（原点位于图像左上角，水平为 x，垂直为 y）下的坐标。
# 转换
x, y = np.meshgrid(range(CAM_WID), range(CAM_HGT))
x = x.astype(np.float32) - CAM_CX
y = y.astype(np.float32) - CAM_CY
img_z = img.copy()

if False:  # 如果需要矫正视线到Z的转换的话使能
    f = (CAM_FX + CAM_FY) / 2.0
    img_z *= f / np.sqrt(x ** 2 + y ** 2 + f ** 2)
pc_x = img_z * x / CAM_FX  # X = Z* (u - cx) / fx
pc_y = img_z * y / CAM_FY  # Y = Z * (v - cy) / fy

# point clouds number: 640*400=256000
pc = np.array([pc_x.ravel(), pc_y.ravel(), img_z.ravel()]).T

# 结果保存
np.savetxt('./data/pointcloud100.csv', pc, fmt='%.18e', delimiter=',', newline='\n')
# 从CSV文件加载点云并显示
pc = np.genfromtxt('./data/pointcloud100.csv', delimiter=',').astype(np.float32) # (256000, 3)

# 可视化点云
x = pc[:, 0]  # x position of point
y = pc[:, 1]  # y position of point
z = pc[:, 2]  # z position of point
# r = temp[:, 3]  # reflectance value of point
d = np.sqrt(x**2 + y**2)  # Map Distance from sensor
vals = 'height'
if vals == 'height':
    col = z
else:
    col = d 
fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
mayavi.mlab.points3d(x, y, z,
                     col, # values used for color
                     mode='point',
                     colormap='spectral', # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0), # used a fixed(r,g,b) instead
                     figure=fig,
                     )
x = np.linspace(5, 5, 50)
y = np.linspace(0, 0, 50)
z = np.linspace(0, 5, 50)
mayavi.mlab.plot3d(x, y, z)
mayavi.mlab.show()


