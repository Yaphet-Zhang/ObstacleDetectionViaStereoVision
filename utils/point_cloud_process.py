import numpy as np
import open3d as o3d
import random
import time
# import mayavi.mlab




def viz_o3d_pc(pc):
    '''
    viz point cloud via open3d
    '''
    o3d.visualization.draw_geometries(pc)




def viz_o3d_pc_video(o3d_vizer, xyz_mesh, box_small_processed, objects_small_processed, box_big, objects_big):
    o3d_vizer.add_geometry(xyz_mesh) # open3d.open3d_pybind.geometry.TriangleMesh
    o3d_vizer.update_geometry(xyz_mesh)
    if len(box_small_processed)!=0: # type: open3d.open3d_pybind.geometry.AxisAlignedBoundingBox
        for box in box_small_processed:
            o3d_vizer.add_geometry(box)
            o3d_vizer.update_geometry(box)
    if len(objects_small_processed)!=0: # type: open3d.open3d_pybind.geometry.PointCloud
        for obj in objects_small_processed:
            o3d_vizer.add_geometry(obj)
            o3d_vizer.update_geometry(obj)
    if len(box_big)!=0: # type: open3d.open3d_pybind.geometry.AxisAlignedBoundingBox
        for box in box_big:
            o3d_vizer.add_geometry(box)
            o3d_vizer.update_geometry(box)
    if len(objects_big)!=0: # type: open3d.open3d_pybind.geometry.PointCloud
        for obj in objects_small_processed:
            o3d_vizer.add_geometry(obj)
            o3d_vizer.update_geometry(obj)

    view_control = o3d_vizer.get_view_control()
    view_control.set_lookat([0, 0.5, -3.5]) # viewpoint position

    o3d_vizer.poll_events()
    o3d_vizer.update_renderer()
    time.sleep(0.01)
    o3d_vizer.clear_geometries()




def np_pc_to_o3d_pc(np_pc):
    '''
    ndarray point cloud -> open3d point cloud
    '''
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(np_pc)
    
    return o3d_pc




def np_to_o3d(np_pc, np_color):
    '''
    ndarray point cloud -> open3d point cloud
    ndarray colors -> open3d colors
    '''
    o3d_color = o3d.geometry.PointCloud()
    o3d_color.points = o3d.utility.Vector3dVector(np_pc) # add point
    o3d_color.colors = o3d.utility.Vector3dVector(np_color) # add color

    return o3d_color




def o3d_pc_to_np_pc(o3d_pc):
    '''
    open3d point cloud -> ndarray point cloud
    '''
    np_pc = np.asarray(o3d_pc.points)
    return np_pc




def calc_outlier(data, factor):
    '''
    find outlier point cloud
    '''
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0, ddof=1) # when data is small, use ddof=1 against skewness
    index_outlier = np.where( np.abs(data - data_mean) > (data_std)*factor )[0] # factor is time of std
    return index_outlier




def depth_to_pc(depth):
    '''
    depth map -> point cloud
    '''
    depth = depth.astype(np.float32)
    # camera parameters (intrinsics)

    # CAM_WID, CAM_HGT = 640, 400
    # CAM_FX, CAM_FY = 795.209, 793.957 # fx，fy 分别为镜头 x，y 方向的焦距（成像平面到镜头的距离）
    # CAM_CX, CAM_CY = 332.031, 231.308 # cx，cy 分别是光心（镜头中心点在成像平面的投影）在图像坐标系（原点位于图像左上角，水平为 x，垂直为 y）下的坐标。

    CAM_WID, CAM_HGT = 1280, 720
    CAM_FX, CAM_FY = 3064.4104, 3064.4104 
    CAM_CX, CAM_CY = 1923.0302, 1078.5358 

    # transfer to mesh
    x, y = np.meshgrid(range(CAM_WID), range(CAM_HGT))
    x = x.astype(np.float32) - CAM_CX
    y = y.astype(np.float32) - CAM_CY
    if False: # 如果需要矫正视线到Z的转换的话使能
        f = (CAM_FX + CAM_FY) / 2.0
        depth *= f / np.sqrt(x ** 2 + y ** 2 + f ** 2)
    pc_x = depth * x / CAM_FX  # X = Z* (u - cx) / fx
    pc_y = depth * y / CAM_FY  # Y = Z * (v - cy) / fy
    # point clouds number: width*height
    pc = np.array([pc_x.ravel(), pc_y.ravel(), depth.ravel()]).T

    return pc




def depth_to_pcl_chatgpt(depth_img, intrinsic, depth_scale=1000.0):
    # 将深度图转换为点云
    height, width = depth_img.shape[:2]
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_img / depth_scale
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]  # X = Z* (u - cx) / fx
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]  # Y = Z * (v - cy) / fy
    pcl = np.stack((x, y, z), axis=-1)
    return pcl




def depth_to_pc_chatgpt(depth):
    # 定义相机内参
    fx, fy = 525.0, 525.0  # 焦距[pix]
    cx, cy = 319.5, 239.5  # 中心点[pix]

    # 获取深度图像的尺寸
    height, width = depth.shape

    # 初始化点云
    point_cloud = np.zeros((height * width, 3))

    # 转换深度图像为点云
    for v in range(height):
        for u in range(width):
            z = depth[v, u] / 1000.0  # 深度图像的单位为毫米，转为米
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point_cloud[v * width + u] = [x, y, z]

    return point_cloud




def viz_inlier_outlier(pc, inlier):
    inlier_pc = pc.select_by_index(inlier) # inlier
    outlier_pc = pc.select_by_index(inlier, invert=True) # outlier

    inlier_pc.paint_uniform_color([0, 0, 0]) # inlier: black
    outlier_pc.paint_uniform_color([1, 0, 0]) # outlier: red
    o3d.visualization.draw_geometries([inlier_pc, outlier_pc])




def calc_array2d_diff(array_0, array_1):
    array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
    array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])
 
    return np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])




def calc_array2d_diff_index(array_0, array_1):
    array_0_rows = array_0.view([('', array_0.dtype)] * array_0.shape[1])
    array_1_rows = array_1.view([('', array_1.dtype)] * array_1.shape[1])

    result = np.setdiff1d(array_0_rows, array_1_rows).view(array_0.dtype).reshape(-1, array_0.shape[1])
    index = np.where(np.isin(array_0_rows, np.setdiff1d(array_0_rows, array_1_rows) ))[0]
    
    return result, index




def cal_color(k):
    ns = []
    while len(ns) < k:
        n = "0x" + "".join([random.choice("01234567789ABCDEF") for j in range(6)])
        if not n in ns:
            ns.append(n)
    return ns






def box_intersection_volume(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    # 计算两个长方体在x轴上的重叠部分
    x_overlap = max(0, min(x1 + w1 / 2, x2 + w2 / 2) - max(x1 - w1 / 2, x2 - w2 / 2))
    
    # 计算两个长方体在y轴上的重叠部分
    y_overlap = max(0, min(y1 + h1 / 2, y2 + h2 / 2) - max(y1 - h1 / 2, y2 - h2 / 2))
    
    # 计算两个长方体在z轴上的重叠部分
    z_overlap = max(0, min(z1 + d1 / 2, z2 + d2 / 2) - max(z1 - d1 / 2, z2 - d2 / 2))
    
    # 计算两个长方体的重叠体积
    overlap_volume = x_overlap * y_overlap * z_overlap
    
    return overlap_volume






# def viz_np_pc_mayavi(pc):
#     '''
#     viz point cloud via mayavi
#     '''
#     x = pc[:, 0]  # x position of point
#     y = pc[:, 1]  # y position of point
#     z = pc[:, 2]  # z position of point
#     # r = temp[:, 3]  # reflectance value of point
#     d = np.sqrt(x**2 + y**2)  # Map Distance from sensor
#     vals = 'height'
#     if vals == 'height':
#         col = z
#     else:
#         col = d 
#     fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
#     mayavi.mlab.points3d(x, y, z,
#                         col, # values used for color
#                         mode='point',
#                         colormap='spectral', # 'bone', 'copper', 'gnuplot'
#                         # color=(0, 1, 0), # used a fixed(r,g,b) instead
#                         figure=fig,
#                         )
#     x = np.linspace(5, 5, 50)
#     y = np.linspace(0, 0, 50)
#     z = np.linspace(0, 5, 50)
#     mayavi.mlab.plot3d(x, y, z)
#     mayavi.mlab.show()