import numpy as np
import cv2
from utils.point_cloud_process import viz_o3d_pc, viz_o3d_pc_video, np_pc_to_o3d_pc, o3d_pc_to_np_pc, calc_outlier, depth_to_pc
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




def calc1(rgb, depth, o3d_vizer=False):

    # plt.imshow(rgb[:, :, ::-1])
    # plt.show()

    ####################
    # 720P (H:720, W:1280)
    HIGHT, WIDTH  = depth.shape
    FX, FY = 3064.4104, 3064.4104 
    CX, CY = 1923.0302, 1078.5358
        
    INTRINSIC_MATRIX = np.array([
        [FX, 0, CX],
        [0, FY, CY],
        [0, 0, 1]
    ]) # 3*3
    EXTRINSIC_MATRIX = np.eye(4) # 4*4




    #################### 2. depth map -> point cloud ####################
    # print('raw pc:', depth.shape[0]*depth.shape[1])
    # print('Depth, max:', depth.max(), 'min', depth.min())

    # add x-y-z axis (open3d)
    xyz_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0]) # 1m = 1000mm
    # np -> o3d
    depth_o3d = o3d.geometry.Image(depth)

    ##### intrinsic & extrinsic #####
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=WIDTH, height=HIGHT, fx=INTRINSIC_MATRIX[0][0], fy=INTRINSIC_MATRIX[1][1], cx=INTRINSIC_MATRIX[0][2], cy=INTRINSIC_MATRIX[1][2])

    # ##### A: depth -> point cloud #####
    # # 1: depth -> point cloud via intrinsic, 2: delet row which are all zero, 3: delet duplicate point cloud, 4: numpy point cloud -> open3d point cloud
    # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_o3d, intrinsic=intrinsic, depth_scale=1000, extrinsic=EXTRINSIC_MATRIX)

    ##### B: rgb-d -> point cloud #####
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    color = o3d.geometry.Image(rgb)
    depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=intrinsic, extrinsic=EXTRINSIC_MATRIX)

    # viz_o3d_pc([xyz_mesh, pcd])


    ##### rotation: pixel -> world #####
    # R = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0]) # rotate 180 degrees around x-axis 
    # pcd = pcd.rotate(R, center=[0, 0, 0]) # rotate
    R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd.transform(R)
    ###
    R_ = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/16, -np.pi/8, 0]) 
    pcd = pcd.rotate(R_, center=[0, 0, 0]) # rotate
    ###

    pcd_np = np.asarray(pcd.points)

    # ##### only use points which less then 10m (z axis) #####
    # z_threshold = -10 # less then 5m
    # z = pcd_np[:, 2]
    # pcd_np = pcd_np[z > z_threshold]
    # pcd = np_pc_to_o3d_pc(pcd_np)

    # print('processed pc:', len(pcd_np))

    x = pcd_np[:, 0]
    y = pcd_np[:, 1]
    # z = pcd_np[:, 2]
    # print('X, max:', x.max(), 'min:', x.min(), 'mean:', x.mean(), 'std:', x.std())
    # print('Y, max:', y.max(), 'min:', y.min(), 'mean:', y.mean(), 'std:', y.std())
    # print('Z, max:', z.max(), 'min:', z.min(), 'mean:', z.mean(), 'std:', z.std())
    # viz_o3d_pc([xyz_mesh, pcd])



    
    #################### 3. downsampling: point cloud -> voxel ####################
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01) # 0.01 -> 1cm
    # viz_o3d_pc([xyz_mesh, np_pc_to_o3d_pc(o3d_pc_to_np_pc(downsampled_pcd))]) # vis via distance
    # viz_o3d_pc([xyz_mesh, downsampled_pcd]) # (raw color)
    # print(downsampled_pcd)




    #################### 4. delete outliers ####################
    inlier_pcd, inlier_index = downsampled_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1) # oak-d: 30, 0.1, AV book: 100, 0.0272, PC book: 20, 2.0
    # inlier_pcd, inlier_index = downsampled_pcd.remove_radius_outlier(nb_points=5, radius=0.02) # 指定半径内，未满指定近邻点数的点，视为异常值
    # viz_inlier_outlier(downsampled_pcd, inlier_index) # viz inlier: black, outlier: red
    # viz_o3d_pc([xyz_mesh, np_pc_to_o3d_pc(o3d_pc_to_np_pc(inlier_pcd))]) # vis via distance
    # viz_o3d_pc([xyz_mesh, inlier_pcd]) # viz inlier result (raw color)
    print(inlier_pcd)




    #################### 5. segment road & obstacle ####################
    #################### 5.1 find peaks
    ##### original #####
    inlier_np = np.asarray(inlier_pcd.points)
    # viz_o3d_pc([ xyz_mesh, inlier_pcd ])
    # print('all', inlier_pcd)

    ##### estimate 2 plane height (y axis) #####
    height_threshold_all = [-0.07539-0.2, -0.07539+0.2] ##### all range
    # inlier_np[:, 1] = inlier_np[:, 1]-0.25+0.07539
    # height_threshold_all = [-0.25-0.2, -0.25+0.2] ##### all range

    height_range = abs(height_threshold_all[0]-height_threshold_all[1])
    # print('range', height_range)
    pcd_low_all = np_pc_to_o3d_pc(inlier_np[ (inlier_np[:, 1] > height_threshold_all[0]) & (inlier_np[:, 1] < height_threshold_all[1]) ])
    # viz_o3d_pc([ xyz_mesh, pcd_low_all ])

    y_height = np.asanyarray(pcd_low_all.points)[:, 1]

    # plot distribution
    n, bins, patches = plt.hist(y_height, bins= int(abs(y_height.max() - y_height.min())*100*3), density=False) # n: 包含每个bin中的数据量或概率的数组, # bins: 包含bin的边界值的数组 (比bin数多一个因为包括左右两边)
    plt.xlabel('Point Cloud Hight [m]')
    plt.ylabel('Frequency')
    plt.title('Distribution')

    # find peaks
    peaks, _ = find_peaks(n, distance=3)
    # plot peaks
    # plt.plot(bins[peaks], n[peaks], 'o')

    # 峰值点对应的bin的数值
    # print('peak value:', bins[peaks])
    # 峰值点对应的bin的数量
    # print('peak num:', n[peaks])
    # 对数组进行排序，并取前两个大的元素
    max_values = sorted(n[peaks], reverse=True)[:2] # big to small
    max_ids = np.argsort(n[peaks])[-2:] # big to small
    # print('最多的两个bin的个数是:', max_values)
    # print('最多的两个bin的索引是:', max_ids)
    estimated_height_few, estimated_height_many = bins[peaks][max_ids]
    # print('估计出来的两个平面的高度:', estimated_height_many, estimated_height_few)
    estimated_height_low1, estimated_height_low2 = min(estimated_height_many, estimated_height_few), max(estimated_height_many, estimated_height_few)

    # 多的一组：当前行驶道路；少的一组：旁边道路
    # plt.annotate('Current', (estimated_height_many, max_values[0]), xytext=None, arrowprops=None) # num: many
    # plt.annotate('Nearby', (estimated_height_few, max_values[1]), xytext=None, arrowprops=None) # num: few
    # plt.show()


    middle_height = abs(estimated_height_low1 - estimated_height_low2)/2

    # 1st low (lowest)
    height_threshold1 = [estimated_height_low1-0.10, estimated_height_low1+middle_height] #####
    # print('range:', height_threshold1)
    low1_id = (inlier_np[:, 1] > height_threshold1[0]) & (inlier_np[:, 1] < height_threshold1[1])
    pcd_low1_np = inlier_np[ low1_id ] # point 
    color_low1_np = np.asarray(inlier_pcd.colors)[low1_id] # color
    pcd_low1 = np_to_o3d(pcd_low1_np, color_low1_np) # point + color
    # pcd_low1 = np_pc_to_o3d_pc(pcd_low1_np) # only point
    # viz_o3d_pc([ xyz_mesh, pcd_low1 ])
    # print('low1', pcd_low1)

    # 2nd low
    # height_threshold2 = [estimated_height_low1, estimated_height_low1+middle_height] ##### for check only 1 plane to RANSAC
    height_threshold2 = [estimated_height_low2-middle_height, estimated_height_low2+0.10] #####
    # print('range:', height_threshold2)
    low2_id = (inlier_np[:, 1] > height_threshold2[0]) & (inlier_np[:, 1] < height_threshold2[1])
    pcd_low2_np = inlier_np[ low2_id ] # point 
    color_low2_np = np.asarray(inlier_pcd.colors)[ low2_id ] # color
    pcd_low2 = np_to_o3d(pcd_low2_np, color_low2_np) # point + color
    # pcd_low2 = np_pc_to_o3d_pc(pcd_low2_np) # only point
    # viz_o3d_pc([ xyz_mesh, pcd_low2 ])
    # print('low2', pcd_low2)


    #################### 5.2 RANSAC (multi: road1 & road2)
    ##### RANSAC #####
    plane_model1, plane_index1 = pcd_low1.segment_plane(distance_threshold=0.01, ransac_n=100, num_iterations=100) # 1cm
    plane_model2, plane_index2 = pcd_low2.segment_plane(distance_threshold=0.01, ransac_n=100, num_iterations=100) # 1cm

    ##### viz estimated plane
    # road 1 (low1/lowest) 
    # a, b, c, d = 0, 0, 1, 0
    [a_road1, b_road1, c_road1, d_road1] = plane_model1
    # print('plane parameters (road 1):', a_road1, b_road1, c_road1, d_road1) # ax + by + cz + d = 0
    xx_road1, zz_road1 = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-10, 0, 0.1))
    yy_road1 = (- a_road1*xx_road1 - c_road1*zz_road1 - d_road1) / b_road1 # according to: ax + by + cz + d = 0
    plane_road1 = np.vstack((xx_road1.ravel(), yy_road1.ravel(), zz_road1.ravel())).T
    # road 2 (low2) 
    [a_road2, b_road2, c_road2, d_road2] = plane_model2
    # print('plane parameters (road 2):', a_road2, b_road2, c_road2, d_road2) # ax + by + cz + d = 0
    xx_road2, zz_road2 = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-10, 0, 0.1))
    yy_road2 = (- a_road2*xx_road2 - c_road2*zz_road2 - d_road2) / b_road2 # according to: ax + by + cz + d = 0
    plane_road2 = np.vstack((xx_road2.ravel(), yy_road2.ravel(), zz_road2.ravel())).T

    ##### distinguish road1, road2, obstacle; and color
    # road 1 -> black
    road_pcd1 = pcd_low1.select_by_index(plane_index1)
    road_pcd1.paint_uniform_color([0, 0, 0])
    print('road 1 (lowest):', road_pcd1)
    # road 2 -> red
    road_pcd2 = pcd_low2.select_by_index(plane_index2)
    road_pcd2.paint_uniform_color([1, 0, 0])
    print('road 2 (low2):', road_pcd2)
    # # outlier (obstacle)
    # obstacle_pcd = pcd_low.select_by_index(plane_index, invert=True)

    # calculate diff of all & (road 1 + road 2) (for get real obstacle) -> blue
    road1_road2_np = np.concatenate((np.asarray(road_pcd1.points), np.asarray(road_pcd2.points)), axis=0)
    obstacle_np, index = calc_array2d_diff_index(np.asarray(inlier_pcd.points), road1_road2_np) # point 
    color_obstacle_np = np.asarray(inlier_pcd.colors)[index] # color 
    obstacle_pcd = np_to_o3d(obstacle_np, color_obstacle_np) # point + color
    obstacle_pcd.paint_uniform_color([0, 0, 1])
    print('obstacle:', obstacle_pcd)

    # viz_o3d_pc([xyz_mesh, road_pcd1, road_pcd2, obstacle_pcd, np_pc_to_o3d_pc(plane_road1), np_pc_to_o3d_pc(plane_road2)])

    # # Tab 1 multi-ground   
    # table1_multi_ground.append([ len(road_pcd1.points), len(road_pcd2.points), len(obstacle_pcd.points) ])


    #################### 5.2 RANSAC (single: road1)
    ##### RANSAC #####
    plane_model3, plane_index3 = inlier_pcd.segment_plane(distance_threshold=0.01, ransac_n=100, num_iterations=100) # 1cm

    ##### viz estimated plane
    # road
    [a_road3, b_road3, c_road3, d_road3] = plane_model3
    # print('plane parameters (road 3):', a_road3, b_road3, c_road3, d_road3) # ax + by + cz + d = 0
    xx_road3, zz_road3 = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-10, 0, 0.1))
    yy_road3 = (- a_road3*xx_road3 - c_road3*zz_road3 - d_road3) / b_road3 # according to: ax + by + cz + d = 0
    plane_road3 = np.vstack((xx_road3.ravel(), yy_road3.ravel(), zz_road3.ravel())).T

    ##### distinguish road1, road2, obstacle; and color
    # road 1 -> black
    road_pcd3 = inlier_pcd.select_by_index(plane_index3)
    road_pcd3.paint_uniform_color([0, 0, 0])
    print('road (single):', road_pcd3)
    # outlier (obstacle) -> blue
    obstacle_pcd_single = inlier_pcd.select_by_index(plane_index3, invert=True)
    obstacle_pcd_single.paint_uniform_color([0, 0, 1])
    print('obstacle (single):', obstacle_pcd_single)

    # viz_o3d_pc([xyz_mesh, road_pcd3, obstacle_pcd_single, np_pc_to_o3d_pc(plane_road3)])

    # # Tab 1 single-ground   
    # table1_single_ground.append([ len(road_pcd3.points), len(obstacle_pcd_single.points) ])




    #################### 6. cluster obstacle ####################
    # # just an example
    # obstacle_pcd = o3d.io.read_point_cloud(r'../data/fragment.pcd') 

    otstacle_np = np.asarray(obstacle_pcd.points).astype(np.float32)

    ##### DBSCAN #####
    # eps: euclidean distance thresold. (if sparse, set big value; if dense, seg small value)
    # min_points: how many points (at least) per 1 cluster 
    # labels: -1: no-cluster (noise)
    labels_small = np.array(obstacle_pcd.cluster_dbscan(eps=0.05, min_points=30, print_progress=False)) # for small object
    labels_big = np.array(obstacle_pcd.cluster_dbscan(eps=0.2, min_points=200, print_progress=False))  # for big object

    # cluster num
    max_label_small = labels_small.max()
    max_label_big = labels_big.max()
    n_clusters_small = max_label_small + 1
    n_clusters_big = max_label_big + 1
    # print('small cluster number (except noise):', n_clusters_small)
    # print('big cluster number (except noise):', n_clusters_big)

    # viz
    colors_small = plt.get_cmap('tab20')(labels_small / max(max_label_small, 1))
    colors_big = plt.get_cmap('tab20')(labels_big / max(max_label_big, 1))

    colors_small[labels_small < 0] = 0 # if noise, set color as black:[0, 0, 0]/white:[1, 1, 1], otherwise [r, g, b]
    colors_big[labels_big < 0] = 0 # if noise, set color as black:[0, 0, 0]/white:[1, 1, 1], otherwise [r, g, b]

    obstacle_pcd.colors = o3d.utility.Vector3dVector(colors_small[:, :3])
    # viz_o3d_pc([xyz_mesh, obstacle_pcd])

    obstacle_pcd.colors = o3d.utility.Vector3dVector(colors_big[:, :3])
    # viz_o3d_pc([xyz_mesh, obstacle_pcd])




    #################### 7. fit each cluster to 3D box ####################
    ##### small 3D BBox (red) #####
    objects_small = []
    box_small = []
    for i in range(n_clusters_small):
        num_pcd = len(otstacle_np[labels_small == i]) # pcd number of 'per object'
        # print('cluster small', i, '->', num_pcd)

        object = np_pc_to_o3d_pc(otstacle_np[labels_small == i]) # pcd of 'per object' 
        aabb = object.get_axis_aligned_bounding_box() ### AABB(轴对其包围盒:Axis-Aligned Bounding Box) 
        aabb.color = (1, 0, 0) # red
        # obb = bottle.get_oriented_bounding_box(robust=True) ### OBB(有向包围盒:Oriented Bounding Box), error
        # obb.color = (0, 1, 0) # green
        objects_small.append(object)
        box_small.append(aabb)


    ##### big 3D BBox (blue) #####
    objects_big = []
    box_big = []
    for i in range(n_clusters_big):
        num_pcd = len(otstacle_np[labels_big == i]) # pcd number of 'per object'
        # print('cluster big', i, '->', num_pcd)

        object = np_pc_to_o3d_pc(otstacle_np[labels_big == i]) # pcd of 'per object' 
        aabb = object.get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1) # blue
        # obb = bottle.get_oriented_bounding_box(robust=True)
        # obb.color = (0, 1, 0) # green
        objects_big.append(object)
        box_big.append(aabb)

    # viz
    # viz_o3d_pc([xyz_mesh] + objects_small + box_small + objects_big + box_big)




    #################### 8. refine 3D box ####################
    if len(box_small) != 0 and len(box_big) != 0: # only both small and big exist
        box_small_processed = []
        objects_small_processed = []

        ids = []
        c = 0
        for box_s in box_small: ##### small
            for box_b in box_big: ##### big
                # viz_o3d_pc([xyz_mesh, box, np_pc_to_o3d_pc(np.array([box.get_center()]))]) # viz box & center
                
                x_1, y_1, z_1 = box_s.get_center() # center coord of x, y, z
                l_1, w_1, h_1 = box_s.get_extent() # length of x, y, z
                box1 = (x_1, y_1, z_1, l_1, w_1, h_1) # 第一个长方体的中心坐标，长宽高

                x_2, y_2, z_2 = box_b.get_center()
                l_2, w_2, h_2 = box_b.get_extent()
                box2 = (x_2, y_2, z_2, l_2, w_2, h_2) # 第二个长方体的中心坐标，长宽高

                ##### calculate intersection (overlap volume)
                v_c = box_intersection_volume(*box1, *box2)

                if v_c > 0: # if overlap only
                    v_1 = l_1 * w_1 * h_1 # volume 1
                    v_2 = l_2 * w_2 * h_2 # volume 2
                    v_small = min(v_1, v_2) # small volume
                    if (v_c / v_small) < (0.3 * 0.3 * 0.3): # only small one when overlap/small is smaller than 0.3 times
                        box_small_processed.append(box_s)
                        objects_small_processed.append(objects_small[c])
                else: # if not overlap
                    box_small_processed.append(box_s)
                    objects_small_processed.append(objects_small[c])
                    # print(objects_small[c])
            c += 1


        ##### viz single frame
        # viz_o3d_pc([xyz_mesh] + box_small_processed + objects_small_processed + box_big + objects_big)

        ##### viz continue frame
        viz_o3d_pc_video(o3d_vizer, xyz_mesh, box_small_processed, objects_small_processed, box_big, objects_big)

    else:
        box_small_processed = box_small
        objects_small_processed = objects_small
        

        ##### viz single frame
        # viz_o3d_pc([xyz_mesh] + box_small_processed + objects_small_processed + box_big + objects_big)

        ##### viz continue frame
        viz_o3d_pc_video(o3d_vizer, xyz_mesh, box_small_processed, objects_small_processed, box_big, objects_big)




if __name__ == "__main__":
    #################### 1. read rgb & left_right & disparity & depth map ####################
    #### 220~399 (180), 1100~1199 (100), 1500~1599 (100), 3900~4099 (200), 4500~4599 (100), (H:720, W:1280) 
    directory_id = '0001'

    # write video (rgb)
    rgb = np.load('../data/{}/rgb{}.npy'.format(directory_id, 220-6))
    h, w, _ = rgb.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cv2_video = cv2.VideoWriter('../data/video/out1.mp4', fourcc, 5, (w*2, h*2)) 

    # show video (point cloud)
    o3d_vizer = o3d.visualization.Visualizer()
    o3d_vizer.create_window(width=w, height=h)

    # table1_multi_ground = [] # road_pcd1, road_pcd2, obstacle_pcd
    # table1_single_ground = [] # road_pcd1_single, obstacle_pcd_single
    for file_id in range(220, 4600):
        if (220 <= file_id <= 399) or (1100 <= file_id <= 1199) or (1500 <= file_id <= 1599) or (3900 <= file_id <= 4099) or (4500 <= file_id <= 4599):
            start_time = time.time()

            # read 
            rgb = np.load('../data/{}/rgb{}.npy'.format(directory_id, file_id-6))
            leftright = np.load('../data/{}/leftright{}.npy'.format(directory_id, file_id))
            disparity = np.load('../data/{}/disparity{}.npy'.format(directory_id, file_id))
            depth = np.load('../data/{}/depth{}.npy'.format(directory_id, file_id))

            # for viz lr
            leftright_ = cv2.cvtColor(leftright, cv2.COLOR_GRAY2BGR) # only copy 1 channel to 3 channels  

            # for viz disparity
            disparityMultiplier = 255 / disparity.max()
            disparity = (disparity * disparityMultiplier).astype(np.uint8) # mono disparity
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET) # color disparity

            # # for viz disparity + rgb
            # blend=cv2.addWeighted(disparity, 0.7, rgb, 0.3, 0)

            print('==================== frame:{} ===================='.format(file_id))
            calc1(rgb, depth, o3d_vizer=o3d_vizer) ##########

            # convert point cloud to screen capture
            img_pc = np.asarray(o3d_vizer.capture_screen_float_buffer())
            img_pc = (img_pc*255).astype(np.uint8)

            img_row1 = np.hstack((rgb, leftright_))
            img_row2 = np.hstack((disparity, img_pc))
            img_combined = np.vstack((img_row1, img_row2))

            cv2_video.write(img_combined)

            end_time = time.time()
            print()
            print()
            print('==================== fps:{} ===================='.format(1 / (end_time-start_time) ))


    # table1_multi_ground = np.array(table1_multi_ground)
    # ave_road1_multi, ave_road2_multi, ave_obstacle_multi = table1_multi_ground[:, 0].mean(), table1_multi_ground[:, 1].mean(), table1_multi_ground[:, 2].mean()
    # print('Multi-RANSAC ||average 1st ground:{}, average 2nd ground:{}, average obstacle:{}'.format(ave_road1_multi, ave_road2_multi, ave_obstacle_multi))

    # table1_single_ground = np.array(table1_single_ground)
    # ave_road1_single, ave_obstacle_single = table1_single_ground[:, 0].mean(), table1_single_ground[:, 1].mean()
    # print('Single-RANSAC || average 1st ground:{}, average obstacle:{}'.format(ave_road1_single, ave_obstacle_single))

    cv2_video.release()
    o3d_vizer.destroy_window()




