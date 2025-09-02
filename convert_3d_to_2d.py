#!/usr/bin/env python

import json
import numpy as np
import cv2
import load_calibration
from scipy.spatial.transform import Rotation as R
import os
from shapely import box, union_all
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# Parameters
ioa_remove_threshold = 0.8 # Intersection over area lower threshold to remove overlapping rectangles
ioa_crop_threshold = 0.2 # Intersection over area lower threshold to crop overlapping rectangles
min_box_area = 100 # Minimum area of a bounding box
min_box_dim = 3 # Minimum length of a bounding box side dimension
box_crop_buffer = 4 # Buffer for soft crop
box_enlarge_dist = 1 # Distance to enlarge bounding box
plot_flag = True
plot_pointcloud = False
plot_rawimg = False
center_crop = False
data_path = "/home/trail/workspace/cadc_devkit/data/cadcd/"
viz_path = "/home/trail/workspace/cadc_devkit/viz/"
if center_crop:
  cam_list = [0, 4]
else:
  cam_list = [0, 1, 2, 3, 4, 5, 6, 7]
select_date_seq = ''
class_list = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Bicycle']
coco_categories = [{'id': num+1, 'name': name} for num, name in enumerate(class_list)]
# max_dist = defaultdict(lambda:100)
# max_dist.update({
#   'Car' : 125,
#   'Truck' : 125,
#   'Bus' : 125,
#   'Pedestrian' : 100,
#   'Bicycle' : 100
# })
# Camera list F, FR, RF, RB, B, LB, LF, FL
max_dist = [100, 100, 60, 60, 100, 60, 60, 100] 
max_dist = {str(cam): max_dist[cam] for cam in [0, 1, 2, 3, 4, 5, 6, 7]}

# CADCD sequences
cadcd = {
    '2018_03_06': [
        '0001','0002','0005','0006','0008','0009','0010',
        '0012','0013','0015','0016','0018'
    ],
    '2018_03_07': [
        '0001','0002','0004','0005','0006','0007'
    ],
    '2019_02_27': [
        '0002','0003', # '0004', removed due to annotation issues
        '0005','0006','0008','0009','0010',
        '0011','0013','0015','0016','0018','0019','0020',
        '0022','0024','0025','0027','0028','0030',
        '0031','0033','0034','0035','0037','0039','0040',
        '0041','0043','0044','0045','0046','0047','0049','0050',
        '0051','0054','0055','0056','0058','0059',
        '0060','0061','0063','0065','0066','0068','0070',
        '0072','0073','0075','0076','0078','0079',
        '0080','0082'
    ]
  }
if select_date_seq:
  cadcd = {select_date_seq[0]: [select_date_seq[1]]}
  print(f"Selected sequence: {cadcd}")
val_set = {
  '2018_03_06/0001',
  '2018_03_06/0008',
  '2018_03_06/0016',
  '2018_03_07/0004',
  '2019_02_27/0009',
  '2019_02_27/0028',
  '2019_02_27/0016',
  '2019_02_27/0033',
  '2019_02_27/0040',
  '2019_02_27/0043',
  '2019_02_27/0054',
  '2019_02_27/0060',
  '2019_02_27/0065',
  '2019_02_27/0076',
}


def cuboids_to_rectangles(cuboids, T_IMG_CAM, T_CAM_LIDAR, img_h, img_w):
  rect_list = []
  for cuboid in cuboids:
    # Get cuboid transformation matrix
    T_Lidar_Cuboid = np.eye(4)
    T_Lidar_Cuboid[0:3,0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix()
    T_Lidar_Cuboid[0][3] = cuboid['position']['x']
    T_Lidar_Cuboid[1][3] = cuboid['position']['y']
    T_Lidar_Cuboid[2][3] = cuboid['position']['z']

    # Get cuboid depth
    T_Cam_Cuboid = np.matmul(T_CAM_LIDAR, T_Lidar_Cuboid)
    cuboid_depth = T_Cam_Cuboid[2][3]
    if cuboid_depth < 0: # Behind camera
      continue

    # Get cuboid dimensions
    width = cuboid['dimensions']['x']
    length = cuboid['dimensions']['y']
    height = cuboid['dimensions']['z']

    # Get cuboid vertices
    front_right_bottom = np.array([[1,0,0,length/2],[0,1,0,-width/2],[0,0,1,-height/2],[0,0,0,1]])
    front_right_top = np.array([[1,0,0,length/2],[0,1,0,-width/2],[0,0,1,height/2],[0,0,0,1]])
    front_left_bottom = np.array([[1,0,0,length/2],[0,1,0,width/2],[0,0,1,-height/2],[0,0,0,1]])
    front_left_top = np.array([[1,0,0,length/2],[0,1,0,width/2],[0,0,1,height/2],[0,0,0,1]])
    back_right_bottom = np.array([[1,0,0,-length/2],[0,1,0,-width/2],[0,0,1,-height/2],[0,0,0,1]])
    back_right_top = np.array([[1,0,0,-length/2],[0,1,0,-width/2],[0,0,1,height/2],[0,0,0,1]])
    back_left_bottom = np.array([[1,0,0,-length/2],[0,1,0,width/2],[0,0,1,-height/2],[0,0,0,1]])
    back_left_top = np.array([[1,0,0,-length/2],[0,1,0,width/2],[0,0,1,height/2],[0,0,0,1]])

    # Transform vertices to camera frame
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_right_bottom))
    if tmp[2][3] < 0:
      continue
    f_r_b = np.matmul(T_IMG_CAM, tmp)
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_right_top))
    if tmp[2][3] < 0:
      continue
    f_r_t = np.matmul(T_IMG_CAM, tmp)
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_left_bottom))
    if tmp[2][3] < 0:
      continue
    f_l_b = np.matmul(T_IMG_CAM, tmp)
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, front_left_top))
    if tmp[2][3] < 0:
      continue
    f_l_t = np.matmul(T_IMG_CAM, tmp)
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_right_bottom))
    if tmp[2][3] < 0:
      continue
    b_r_b = np.matmul(T_IMG_CAM, tmp)
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_right_top))
    if tmp[2][3] < 0:
      continue
    b_r_t = np.matmul(T_IMG_CAM, tmp)
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_left_bottom))
    if tmp[2][3] < 0:
      continue
    b_l_b = np.matmul(T_IMG_CAM, tmp)
    tmp = np.matmul(T_CAM_LIDAR, np.matmul(T_Lidar_Cuboid, back_left_top))
    if tmp[2][3] < 0:
      continue
    b_l_t = np.matmul(T_IMG_CAM, tmp)

    # Project vertices to image
    f_r_b_coord = (int(f_r_b[0][3]/f_r_b[2][3]), int(f_r_b[1][3]/f_r_b[2][3]))
    f_r_t_coord = (int(f_r_t[0][3]/f_r_t[2][3]), int(f_r_t[1][3]/f_r_t[2][3]))
    f_l_b_coord = (int(f_l_b[0][3]/f_l_b[2][3]), int(f_l_b[1][3]/f_l_b[2][3]))
    f_l_t_coord = (int(f_l_t[0][3]/f_l_t[2][3]), int(f_l_t[1][3]/f_l_t[2][3]))
    b_r_b_coord = (int(b_r_b[0][3]/b_r_b[2][3]), int(b_r_b[1][3]/b_r_b[2][3]))
    b_r_t_coord = (int(b_r_t[0][3]/b_r_t[2][3]), int(b_r_t[1][3]/b_r_t[2][3]))
    b_l_b_coord = (int(b_l_b[0][3]/b_l_b[2][3]), int(b_l_b[1][3]/b_l_b[2][3]))
    b_l_t_coord = (int(b_l_t[0][3]/b_l_t[2][3]), int(b_l_t[1][3]/b_l_t[2][3]))

    # Check if vertices out of bounds of image and crop
    oob_frb = f_r_b_coord[0] < 0 or f_r_b_coord[0] > img_w or f_r_b_coord[1] < 0 or f_r_b_coord[1] > img_h
    oob_frt = f_r_t_coord[0] < 0 or f_r_t_coord[0] > img_w or f_r_t_coord[1] < 0 or f_r_t_coord[1] > img_h
    oob_flb = f_l_b_coord[0] < 0 or f_l_b_coord[0] > img_w or f_l_b_coord[1] < 0 or f_l_b_coord[1] > img_h
    oob_flt = f_l_t_coord[0] < 0 or f_l_t_coord[0] > img_w or f_l_t_coord[1] < 0 or f_l_t_coord[1] > img_h
    oob_brb = b_r_b_coord[0] < 0 or b_r_b_coord[0] > img_w or b_r_b_coord[1] < 0 or b_r_b_coord[1] > img_h
    oob_brt = b_r_t_coord[0] < 0 or b_r_t_coord[0] > img_w or b_r_t_coord[1] < 0 or b_r_t_coord[1] > img_h
    oob_blb = b_l_b_coord[0] < 0 or b_l_b_coord[0] > img_w or b_l_b_coord[1] < 0 or b_l_b_coord[1] > img_h
    oob_blt = b_l_t_coord[0] < 0 or b_l_t_coord[0] > img_w or b_l_t_coord[1] < 0 or b_l_t_coord[1] > img_h
    oob_sum = oob_frb + oob_frt + oob_flb + oob_flt + oob_brb + oob_brt + oob_blb + oob_blt
    if oob_sum > 4:
      continue
    if oob_frb:
      f_r_b_coord = (max(0, f_r_b_coord[0]), max(0, f_r_b_coord[1]))
    if oob_frt:
      f_r_t_coord = (max(0, f_r_t_coord[0]), max(0, f_r_t_coord[1]))
    if oob_flb:
      f_l_b_coord = (max(0, f_l_b_coord[0]), max(0, f_l_b_coord[1]))
    if oob_flt:
      f_l_t_coord = (max(0, f_l_t_coord[0]), max(0, f_l_t_coord[1]))
    if oob_brb:
      b_r_b_coord = (max(0, b_r_b_coord[0]), max(0, b_r_b_coord[1]))
    if oob_brt:
      b_r_t_coord = (max(0, b_r_t_coord[0]), max(0, b_r_t_coord[1]))
    if oob_blb:
      b_l_b_coord = (max(0, b_l_b_coord[0]), max(0, b_l_b_coord[1]))
    if oob_blt:
      b_l_t_coord = (max(0, b_l_t_coord[0]), max(0, b_l_t_coord[1]))
    
    # Get bounding box rectangle
    r_coord = max(f_r_b_coord[0], f_r_t_coord[0], f_l_b_coord[0], f_l_t_coord[0], b_r_b_coord[0], b_r_t_coord[0], b_l_b_coord[0], b_l_t_coord[0])
    l_coord = min(f_r_b_coord[0], f_r_t_coord[0], f_l_b_coord[0], f_l_t_coord[0], b_r_b_coord[0], b_r_t_coord[0], b_l_b_coord[0], b_l_t_coord[0])
    t_coord = min(f_r_b_coord[1], f_r_t_coord[1], f_l_b_coord[1], f_l_t_coord[1], b_r_b_coord[1], b_r_t_coord[1], b_l_b_coord[1], b_l_t_coord[1])
    b_coord = max(f_r_b_coord[1], f_r_t_coord[1], f_l_b_coord[1], f_l_t_coord[1], b_r_b_coord[1], b_r_t_coord[1], b_l_b_coord[1], b_l_t_coord[1])
    l_t_coord = (l_coord, t_coord)
    r_b_coord = (r_coord, b_coord)
    rect_list.append((l_t_coord, r_b_coord, cuboid_depth, cuboid['label'], cuboid['uuid']))

  return rect_list


def remove_overlap_rects(rect_list):
  remove_list = []
  crop_list = []
  for i in range(len(rect_list)):
    intersection_box_list = []
    box_i = box(rect_list[i][0][0], rect_list[i][0][1], rect_list[i][1][0], rect_list[i][1][1])
    box_i_depth = rect_list[i][2]
    box_i_visible_region = box_i
    
    # Compute intersection with other rectangles
    for j in range(len(rect_list)):
      box_j = box(rect_list[j][0][0]-box_enlarge_dist, rect_list[j][0][1]-box_enlarge_dist, 
                  rect_list[j][1][0]+box_enlarge_dist, rect_list[j][1][1]+box_enlarge_dist)
      box_j_depth = rect_list[j][2]
      if i == j or box_i_depth < box_j_depth:
        continue
      intesection = box_i.intersection(box_j)
      if intesection.area == 0:
        continue
      intersection_box_list.append(intesection)
      box_i_visible_region = box_i_visible_region.difference(box_j)
    intersection_polygon = union_all(intersection_box_list)

    # Remove and crop boxes
    if intersection_polygon.area > 0:
      box_i_ioa = intersection_polygon.area / box_i.area
      if box_i_ioa > ioa_remove_threshold:
        remove_list.append(i)
      elif box_i_ioa > ioa_crop_threshold:
        minx, miny, maxx, maxy = box_i_visible_region.bounds

        # Soft crop
        if minx != rect_list[i][0][0]:
          minx = max(minx - box_crop_buffer, rect_list[i][0][0])
        if miny != rect_list[i][0][1]:
          miny = max(miny - box_crop_buffer, rect_list[i][0][1])
        if maxx != rect_list[i][1][0]:
          maxx = min(maxx + box_crop_buffer, rect_list[i][1][0])
        if maxy != rect_list[i][1][1]:
          maxy = min(maxy + box_crop_buffer, rect_list[i][1][1])

        l_t_coord = (int(minx), int(miny))
        r_b_coord = (int(maxx), int(maxy))
        if l_t_coord[0] == rect_list[i][0][0] and l_t_coord[1] == rect_list[i][0][1] \
          and r_b_coord[0] == rect_list[i][1][0] and r_b_coord[1] == rect_list[i][1][1]:
          continue
        crop_list.append(i)
        rect_list[i] = (l_t_coord, r_b_coord, rect_list[i][2], rect_list[i][3], rect_list[i][4])

  return remove_list, crop_list, rect_list        


def convert_rect_to_coco(coco_dict, rect_list):
  for rect in rect_list:
    # Unpack rectangle
    x_min = rect[0][0]
    y_min = rect[0][1]
    x_max = rect[1][0]
    y_max = rect[1][1]
    class_name = rect[3]
    track_str = rect[4]

    # Get class id and track id
    if class_name not in [cat['name'] for cat in coco_dict['categories']]:
      raise ValueError(f"Class {class_name} not in categories")
    else:
      class_id = [cat['id'] for cat in coco_dict['categories'] if cat['name'] == class_name][0]
    if track_str not in [inst['name'] for inst in coco_dict['instances']]:
      track_id = len(coco_dict['instances'])+1
      coco_dict['instances'].append({'id': track_id, 'name': track_str})
    else:
      track_id = [inst['id'] for inst in coco_dict['instances'] if inst['name'] == track_str][0]

    # Pack annotation
    ann_count = len(coco_dict['annotations'])
    img_count = len(coco_dict['images'])
    video_count = len(coco_dict['videos'])
    data_anno = dict(
      id=ann_count+1,
      image_id=img_count,
      video_id=video_count,
      category_id=class_id,
      instance_id=track_id,
      bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
      area=(x_max - x_min) * (y_max - y_min),
      segmentation=[(0, 0)],
      iscrowd=0)
    coco_dict['annotations'].append(data_anno)

  return coco_dict

def plot_bev(s1,s2,f1,f2,frame,lidar_path,annotations_path,save_path,im_px):
    '''
    :param s1: example 15 (15 meter to the left of the car)
    :param s2: s2 meters from the right of the car
    :param f1: f1 meters from the front of the car
    :param f2: f2 meters from the back of the car
    :param frame: the frame number
    :return:
    '''
    #limit the viewing range
    side_range = [-s1,s2] #15 meters from either side of the car
    fwd_range = [-f1,f2] # 15 m infront of the car
    scan_data = np.fromfile(lidar_path, dtype= np.float32) #numpy from file reads binary file
    #scan_data is a single row of all the lidar values
    # 2D array where each row contains a point [x, y, z, intensity]
    #we covert scan_data to format said above
    lidar = scan_data.reshape((-1, 4));
    lidar_x = lidar[:,0]
    lidar_y = lidar[:,1]
    lidar_z = lidar [:,2]
    lidar_x_trunc = []
    lidar_y_trunc = []
    lidar_z_trunc = []

    for i in range(len(lidar_x)):
        if lidar_x[i] > fwd_range[0] and lidar_x[i] < fwd_range[1]: #get the lidar coordinates
            if lidar_y[i] > side_range[0] and lidar_y[i] < side_range[1]:
                lidar_x_trunc.append(lidar_x[i])
                lidar_y_trunc.append(lidar_y[i])
                lidar_z_trunc.append(lidar_z[i])

    # to use for the plot
    x_img = [i* -1 for i in lidar_y_trunc] #in the image plot, the negative lidar y axis is the img x axis
    y_img = lidar_x_trunc #the lidar x axis is the img y axis
    pixel_values = lidar_z_trunc

    #shift values such that 0,0 is the minimum
    x_img = [i -side_range[0] for i in x_img]
    y_img = [i -fwd_range[0] for i in y_img]

    '''
    tracklets 
    '''
    # Load 3d annotations
    annotations_data = None
    with open(annotations_path) as f:
        annotations_data = json.load(f)

    # Add each cuboid to image
    '''
    Rotations in 3 dimensions can be represented by a sequece of 3 rotations around a sequence of axes. 
    In theory, any three axes spanning the 3D Euclidean space are enough. In practice the axes of rotation are chosen to be the basis vectors.
    The three rotations can either be in a global frame of reference (extrinsic) or in a body centred frame of refernce (intrinsic),
    which is attached to, and moves with, the object under rotation
    '''
    # PLOT THE IMAGE
    cmap = "jet"    # Color map to use
    dpi = 96       # Image resolution
    x_max = side_range[1] - side_range[0]
    y_max = fwd_range[1] - fwd_range[0]
    fig, ax = plt.subplots(figsize=(im_px/dpi, im_px/dpi), dpi=dpi)

    # the coordinates in the tracklet json are lidar coords
    x_trunc = []
    y_trunc = []
    x_1 = []
    x_2 =[]
    x_3 = []
    x_4 =[]
    y_1 =[]
    y_2 =[]
    y_3 = []
    y_4 =[]

    for cuboid in annotations_data[frame]['cuboids']:
        T_Lidar_Cuboid = np.eye(4);  # identify matrix
        T_Lidar_Cuboid[0:3, 0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix();  # rotate the identity matrix
        T_Lidar_Cuboid[0][3] = cuboid['position']['x'];  # center of the tracklet, from cuboid to lidar
        T_Lidar_Cuboid[1][3] = cuboid['position']['y'];
        T_Lidar_Cuboid[2][3] = cuboid['position']['z'];

        if cuboid['position']['x']> fwd_range[0] and cuboid['position']['x'] < fwd_range[1]: #make sure the cuboid is within the range we want to see
            if cuboid['position']['y'] > side_range[0] and cuboid['position']['y'] < side_range[1]:
                x_trunc.append(cuboid['position']['x'])
                y_trunc.append(cuboid['position']['y'])
                width = cuboid['dimensions']['x'];
                length = cuboid['dimensions']['y'];
                height = cuboid['dimensions']['z'];
                radius = 3
                #the top view of the tracklet in the "cuboid frame". The cuboid frame is a cuboid with origin (0,0,0)
                #we are making a cuboid that has the dimensions of the tracklet but is located at the origin
                front_right_top = np.array(
                    [[1, 0, 0, length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);
                front_left_top = np.array(
                    [[1, 0, 0, length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);
                back_right_top = np.array(
                    [[1, 0, 0, -length / 2], [0, 1, 0, width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);
                back_left_top = np.array(
                    [[1, 0, 0, -length / 2], [0, 1, 0, -width / 2], [0, 0, 1, height / 2], [0, 0, 0, 1]]);

                # Project to lidar
                f_r_t =  np.matmul(T_Lidar_Cuboid, front_right_top)
                f_l_t  = np.matmul(T_Lidar_Cuboid, front_left_top)
                b_r_t  = np.matmul(T_Lidar_Cuboid, back_right_top)
                b_l_t = np.matmul(T_Lidar_Cuboid, back_left_top)
                x_1.append(f_r_t[0][3])
                y_1.append(f_r_t[1][3])
                x_2.append(f_l_t[0][3])
                y_2.append(f_l_t[1][3])
                x_3.append(b_r_t[0][3])
                y_3.append(b_r_t[1][3])
                x_4.append(b_l_t[0][3])
                y_4.append(b_l_t[1][3])

    # to use for the plot
    x_img_tracklets = [i * -1 for i in y_trunc]  # in the image to plot, the negative lidar y axis is the img x axis
    y_img_tracklets = x_trunc  # the lidar x axis is the img y axis
    x_img_1 = [i * -1 for i in y_1]
    y_img_1 = x_1
    x_img_2 = [i * -1 for i in y_2]
    y_img_2 = x_2
    x_img_3 = [i * -1 for i in y_3]
    y_img_3 = x_3
    x_img_4 = [i * -1 for i in y_4]
    y_img_4 = x_4

    # shift values such that 0,0 is the minimum
    x_img_tracklets = [i -side_range[0] for i in x_img_tracklets]
    y_img_tracklets = [i -fwd_range[0] for i in y_img_tracklets]
    x_img_1 = [i -side_range[0] for i in x_img_1]
    y_img_1 = [i - fwd_range[0] for i in y_img_1]
    x_img_2 = [i -side_range[0] for i in x_img_2]
    y_img_2 = [i - fwd_range[0] for i in y_img_2]
    x_img_3 = [i -side_range[0] for i in x_img_3]
    y_img_3 = [i - fwd_range[0] for i in y_img_3]
    x_img_4 = [i -side_range[0] for i in x_img_4]
    y_img_4 = [i - fwd_range[0] for i in y_img_4]

    for i in range(len(x_img_1)): #plot the tracklets
        poly = np.array([[x_img_1[i], y_img_1[i]], [x_img_2[i], y_img_2[i]], [x_img_4[i], y_img_4[i]], [x_img_3[i], y_img_3[i]]])
        polys = patches.Polygon(poly,closed=True,fill=False, edgecolor ='r', linewidth=1)
        ax.add_patch(polys)

    # ax.scatter(x_img_tracklets,y_img_tracklets, marker ='o', color='red', linewidths=1) #center of the tracklets
    ax.scatter(x_img, y_img, s=1, c=pixel_values, alpha=1.0, cmap=cmap)
    ax.set_facecolor((0, 0, 0))  # backgrounD is black
    ax.axis('scaled')  # {equal, scaled}
    ax.xaxis.set_visible(False)  # Do not draw axis tick marks
    ax.yaxis.set_visible(False)  # Do not draw axis tick marks
    plt.xlim([0, x_max])
    plt.ylim([0, y_max])  
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)


def project_points(img, lidar_path, T_IMG_CAM, T_CAM_LIDAR, dist_coeffs, DISTORTED):
  scan_data = np.fromfile(lidar_path, dtype=np.float32)

  # 2D array where each row contains a point [x, y, z, intensity]
  lidar = scan_data.reshape((-1, 4))

  # Get height and width of the image
  h, w = img.shape[:2]
  projected_points = []
  [rows, cols] = lidar.shape

  for i in range(rows):
    p = np.array([0.0, 0.0, 0.0, 1.0])
    p[0:3] = lidar[i,0:3]
    projected_p =  np.matmul(T_CAM_LIDAR, p.transpose())
    if projected_p[2] < 2: # arbitrary cut off
      continue
    projected_points.append([projected_p[0], projected_p[1], projected_p[2]])

  # Project points to image
  projected_points_np = np.array(projected_points)
  image_points = []
  [rows, cols] = projected_points_np.shape
  for i in range(rows):
    point = np.array([0.0, 0.0, 0.0, 1.0])
    point[0:3] = projected_points_np[i]
    if DISTORTED:
      rvec = tvec = np.zeros(3)
      image_points, jac = cv2.projectPoints(np.array([point[0:3]]), rvec, tvec, T_IMG_CAM[0:3,0:3], dist_coeffs)
      image_points.append([image_points[0,0,0], image_points[0,0,1]])
    else:
      curr = np.matmul(T_IMG_CAM, point.transpose()).transpose()
      done = [curr[0] / curr[2], curr[1] / curr[2]]
      image_points.append(done)

  radius = 0
  [rows, cols] = projected_points_np.shape

  NUM_COLOURS = 7
  rainbow = [
    [0, 0, 255], # Red
    [0, 127, 255], # Orange
    [0, 255, 255], # Yellow
    [0, 255, 0], # Green
    [255, 0, 0], # Blue
    [130, 0, 75], # Indigo
    [211, 0, 148] # Violet
  ]

  for i in range(rows):
    colour = int(NUM_COLOURS*(projected_points_np[i][2]/70))
    x = int(image_points[i][0])
    y = int(image_points[i][1])
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
      continue
    if colour > NUM_COLOURS-1:
      continue
    cv2.circle(img, (x,y), radius, rainbow[colour], thickness=2, lineType=8, shift=0)

  return img


##################################### MAIN #####################################

def main():
  coco_dict_all = dict(
            categories=coco_categories,
            videos=[],
            images=[],
            annotations=[],
            instances=[])
  coco_dict_train = dict(
            categories=coco_categories,
            videos=[],
            images=[],
            annotations=[],
            instances=[])
  coco_dict_val = dict(
            categories=coco_categories,
            videos=[],
            images=[],
            annotations=[],
            instances=[])
  cam_seq_trainval_list = []
  class_counts = {}
  total_seq = sum([len(cadcd[date]) for date in cadcd])
  for cam in cam_list:
    pbar = tqdm(total=total_seq, desc=f"Processing cam {cam}")
    cam = str(cam)
    viz_filepaths = []
    for date in cadcd:
      for seq in cadcd[date]:
        pbar.set_description(f"Processing cam {cam} ({date}_{seq})")
        timestamps = data_path + date + "/" + seq + "/labeled/image_0" + cam + "/timestamps.txt"
        seq_split = 'val' if date + "/" + seq in val_set else 'train'
        with open(timestamps) as f:
          num_timestamps = sum(1 for line in f)
        frame_id = 0
        for frame in range(num_timestamps):
          frame_id += 1
          calib_path = data_path + date + "/" + "calib/"
          img_path = data_path + date + "/" + seq + "/labeled/image_0" + cam + "/data/" + format(frame, '010') + ".png"
          annotations_file = data_path + date + "/" + seq + "/3d_ann.json"

          # Load 3d annotations
          annotations_data = None
          with open(annotations_file) as f:
            annotations_data = json.load(f)
          calib = load_calibration.load_calibration(calib_path)

          # Projection matrix from camera to image frame
          T_IMG_CAM = np.eye(4)
          T_IMG_CAM[0:3,0:3] = np.array(calib['CAM0' + cam]['camera_matrix']['data']).reshape(-1, 3)
          T_IMG_CAM = T_IMG_CAM[0:3,0:4] # remove last row
          T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + cam]))

          # Add each cuboid to image
          img_w = 1280
          img_h = 1024
          cuboids = annotations_data[frame]['cuboids']
          rect_list = cuboids_to_rectangles(cuboids, T_IMG_CAM, T_CAM_LIDAR, img_h, img_w)

          # Center crop image and rectangles
          if plot_flag:
            img = cv2.imread(img_path)
            img_raw = img.copy()
            if center_crop:
              img = img[int(img_h/4):int(3*img_h/4), int(img_w/4):int(3*img_w/4), :]
              img_raw_crop = img.copy()
          if center_crop:
            for i in range(len(rect_list)):
              l_coord = rect_list[i][0][0] - int(img_w/4)
              t_coord = rect_list[i][0][1] - int(img_h/4)
              r_coord = rect_list[i][1][0] - int(img_w/4)
              b_coord = rect_list[i][1][1] - int(img_h/4)
              if (l_coord < 0 or l_coord > img_w/2) and (r_coord < 0 or r_coord > img_w/2) \
                and (t_coord < 0 or t_coord > img_h/2) and (b_coord < 0 or b_coord > img_h/2):
                rect_list[i] = None
                continue
              l_coord = min(max(0, l_coord), int(img_w/2))
              t_coord = min(max(0, t_coord), int(img_h/2))
              r_coord = min(max(0, r_coord), int(img_w/2))
              b_coord = min(max(0, b_coord), int(img_h/2))
              rect_list[i] = ((l_coord, t_coord), (r_coord, b_coord), rect_list[i][2], rect_list[i][3], rect_list[i][4])
            rect_list = [rect for rect in rect_list if rect is not None]
            img_w = int(img_w/2)
            img_h = int(img_h/2)
          assert img_w == img.shape[1] and img_h == img.shape[0]

          # Remove overlapping rectangles
          remove_occ_list, crop_list, rect_list = remove_overlap_rects(rect_list)
          
          # Remove small rectangles
          remove_dist_list = []
          for i in range(len(rect_list)):
            if i in remove_occ_list:
              continue
            min_box_side = min(rect_list[i][1][0] - rect_list[i][0][0], rect_list[i][1][1] - rect_list[i][0][1])
            box_area = (rect_list[i][1][0] - rect_list[i][0][0]) * (rect_list[i][1][1] - rect_list[i][0][1])
            if box_area <= min_box_area or min_box_side <= min_box_dim or rect_list[i][2] > max_dist[cam]:
              remove_dist_list.append(i)
          
          # Remove class list
          remove_class_list = []
          for i in range(len(rect_list)):
            if rect_list[i][3] not in class_list:
              remove_class_list.append(i)

          remove_list = remove_occ_list + remove_dist_list + remove_class_list

          # Plot rectangles and text
          if plot_flag:
            # for i in remove_occ_list:
            #   cv2.rectangle(img, rect_list[i][0], rect_list[i][1], [255, 0, 0], thickness=1, lineType=8, shift=0)
            #   text = rect_list[i][3][:3] + str(int(rect_list[i][2]))
            #   cv2.putText(img, text, rect_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
            for i in remove_dist_list:
              if i not in remove_class_list:
                cv2.rectangle(img, rect_list[i][0], rect_list[i][1], [255, 255, 255], thickness=1, lineType=8, shift=0)
                text = rect_list[i][3][:3] + str(int(rect_list[i][2]))
                cv2.putText(img, text, rect_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            for i in crop_list:
              if i not in remove_list:
                cv2.rectangle(img, rect_list[i][0], rect_list[i][1], [0, 0, 255], thickness=1, lineType=8, shift=0)
                text = rect_list[i][3][:3] + str(int(rect_list[i][2]))
                cv2.putText(img, text, rect_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
            for i in range(len(rect_list)):
              if i not in remove_occ_list and i not in remove_dist_list and i not in crop_list and i not in remove_class_list:
                cv2.rectangle(img, rect_list[i][0], rect_list[i][1], [0, 255, 0], thickness=1, lineType=8, shift=0)
                text = rect_list[i][3][:3] + str(int(rect_list[i][2]))
                cv2.putText(img, text, rect_list[i][0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
            text = "cam" + cam + "_" + date + "_" + seq + "_" + str(frame)
            if center_crop:
              text += "_crop"
            cv2.putText(img, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            
            if plot_pointcloud:
              assert center_crop and plot_rawimg
              # Plot point cloud on image
              img_pc = img_raw.copy()
              dist_coeffs = np.array(calib['CAM0' + cam]['distortion_coefficients']['data'])
              lidar_path = data_path + date + "/" + seq + "/labeled/lidar_points/data/" + format(frame, '010') + ".bin"
              DISTORTED = False
              img_pc = project_points(img_pc, lidar_path, T_IMG_CAM, T_CAM_LIDAR, dist_coeffs, DISTORTED)
              if center_crop:
                img_pc = img_pc[int(img_h/2):int(3*img_h/2), int(img_w/2):int(3*img_w/2), :]

              # Plot BEV
              bev_path = viz_path + "bev_temp.png"
              os.makedirs(os.path.dirname(bev_path), exist_ok=True)
              bev_lim_fb = 80
              bev_lim_lr = bev_lim_fb*1.25
              im_px = 827 # needs to be tuned to ge matching dimensions based on dpi
              plot_bev(bev_lim_lr, bev_lim_lr, bev_lim_fb, bev_lim_fb, frame, lidar_path, annotations_file, bev_path, im_px)
              img_bev = cv2.imread(bev_path)
              # print(img_pc.shape, img_bev.shape)
              img_pc = cv2.hconcat([img_pc, img_bev])

            # Save image
            if center_crop:
              viz_filepath = viz_path + "cam" + cam + "_crop/" + date + "_" + seq + "_" + str(frame) + ".png"
            else:
              viz_filepath = viz_path + "cam" + cam + "/" + date + "_" + seq + "_" + str(frame) + ".png"
            if plot_rawimg:
              if center_crop:
                img = cv2.hconcat([img_raw_crop, img])
              else:
                img = cv2.hconcat([img_raw, img])

            viz_filepaths.append(viz_filepath)
            if plot_pointcloud:
              img = cv2.vconcat([img, img_pc])
            os.makedirs(os.path.dirname(viz_filepath), exist_ok=True)
            cv2.imwrite(viz_filepath, img)
            
          # Convert to COCO format
          rect_list = [rect_list[i] for i in range(len(rect_list)) if i not in remove_list]
          seq_rel_path = date + "/" + seq + "/labeled/image_0" + cam
          img_rel_path = seq_rel_path + "/data/" + format(frame, '010') + ".png"
          seq_count = len(coco_dict_all['videos'])
          img_count = len(coco_dict_all['images'])
          if seq_rel_path not in [vid['name'] for vid in coco_dict_all['videos']]:
            coco_dict_all['videos'].append(
              dict(id=seq_count+1, name=seq_rel_path, fps=1/0.3, height=img_h, width=img_w))          
          coco_dict_all['images'].append(
            dict(id=img_count+1, video_id=seq_count+1, frame_id=frame_id, file_name=img_rel_path, height=img_h, width=img_w))
          coco_dict_all = convert_rect_to_coco(coco_dict_all, rect_list)
          if seq_split == 'train':
            seq_count = len(coco_dict_train['videos'])
            img_count = len(coco_dict_train['images'])
            if seq_rel_path not in [vid['name'] for vid in coco_dict_train['videos']]:
              coco_dict_train['videos'].append(
                dict(id=seq_count+1, name=seq_rel_path, fps=1/0.3, height=img_h, width=img_w))          
            coco_dict_train['images'].append(
              dict(id=img_count+1, video_id=seq_count+1, frame_id=frame_id, file_name=img_rel_path, height=img_h, width=img_w))
            coco_dict_train = convert_rect_to_coco(coco_dict_train, rect_list)
          else:
            seq_count = len(coco_dict_val['videos'])
            img_count = len(coco_dict_val['images'])
            if seq_rel_path not in [vid['name'] for vid in coco_dict_val['videos']]:
              coco_dict_val['videos'].append(
                dict(id=seq_count+1, name=seq_rel_path, fps=1/0.3, height=img_h, width=img_w))          
            coco_dict_val['images'].append(
              dict(id=img_count+1, video_id=seq_count+1, frame_id=frame_id, file_name=img_rel_path, height=img_h, width=img_w))
            coco_dict_val = convert_rect_to_coco(coco_dict_val, rect_list)

          # Update class counts
          for rect in rect_list:
            class_name = rect[3]
            if class_name not in class_counts:
              class_counts[class_name] = 1
            else:
              class_counts[class_name] += 1
          img_count += 1

        # End of frame loop
        pbar.update(1)
        cam_seq_trainval_list.append({'seq': date + "/" + seq + "/cam" + cam, 'split': seq_split})

      # End of seq loop
      # break # For debugging purposes

    # End of date loop
    pbar.close()

    # Create video from images for single camera
    if plot_flag:
      print(f">> Creating video from images from cam {cam}")
      video_path = viz_path + "cam" + cam + ".mp4"
      if center_crop:
        video_path = viz_path + "cam" + cam + "_crop.mp4"
      os.makedirs(os.path.dirname(video_path), exist_ok=True)
      fps = 5
      img = cv2.imread(viz_filepaths[0])
      height, width, layers = img.shape
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'MJPG', etc.
      video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
      for path in viz_filepaths:
          img = cv2.imread(path)
          video.write(img)
      video.release()
      print(f'Video saved to {video_path}')
      
    # break # For debugging purposes

  # End of cam loop

  # Write COCO format to file
  print(">> Writing COCO format to file")
  out_file = data_path + "cadc_coco_2d_ann_all.json"
  if center_crop:
    out_file = data_path + "cadc_coco_2d_ann_all_crop.json"
  os.makedirs(os.path.dirname(out_file), exist_ok=True)
  with open(out_file, 'w') as out_file:
    json.dump(coco_dict_all, out_file)
  print(f'Annotations saved to {out_file}')

  out_file = data_path + "cadc_coco_2d_ann_train.json"
  if center_crop:
    out_file = data_path + "cadc_coco_2d_ann_train_crop.json"
  os.makedirs(os.path.dirname(out_file), exist_ok=True)
  with open(out_file, 'w') as out_file:
    json.dump(coco_dict_train, out_file)
  print(f'Annotations saved to {out_file}')

  out_file = data_path + "cadc_coco_2d_ann_val.json"
  if center_crop:
    out_file = data_path + "cadc_coco_2d_ann_val_crop.json"
  os.makedirs(os.path.dirname(out_file), exist_ok=True)
  with open(out_file, 'w') as out_file:
    json.dump(coco_dict_val, out_file)
  print(f'Annotations saved to {out_file}')
      
  # Write trainval split to file
  out_file = data_path + "cadc_trainval_split.json"
  if center_crop:
    out_file = data_path + "cadc_trainval_split_crop.json"
  os.makedirs(os.path.dirname(out_file), exist_ok=True)
  with open(out_file, 'w') as out_file:
    json.dump(cam_seq_trainval_list, out_file)
    print(f'Trainval split saved to {out_file}')
  
  # Stats to file
  print(">> Done")
  text_to_print = f"Number of videos: {len(coco_dict_all['videos'])} ({len(coco_dict_train['videos'])} train, {len(coco_dict_val['videos'])} val)\n"
  text_to_print += f"Number of images: {len(coco_dict_all['images'])} ({len(coco_dict_train['images'])} train, {len(coco_dict_val['images'])} val)\n"
  text_to_print += f"Number of annotations: {len(coco_dict_all['annotations'])} ({len(coco_dict_train['annotations'])} train, {len(coco_dict_val['annotations'])} val)\n"
  text_to_print += f"Number of instances: {len(coco_dict_all['instances'])} ({len(coco_dict_train['instances'])} train, {len(coco_dict_val['instances'])} val)\n"
  text_to_print += f"Number of classes: {len(coco_dict_all['categories'])} ({len(coco_dict_train['categories'])} train, {len(coco_dict_val['categories'])} val)\n"
  text_to_print += f"Train/Val split: {len(coco_dict_train['images']) / (len(coco_dict_train['images']) + len(coco_dict_val['images']))}\n"
  text_to_print += "Class counts:\n"
  text_to_print += json.dumps(class_counts, indent=2)
  print(text_to_print)
  stats_path = data_path + "cadc_coco_2d_ann_stats.txt"
  if center_crop:
    stats_path = data_path + "cadc_coco_2d_ann_stats_crop.txt"
  with open(stats_path, "w") as f:
    f.write(text_to_print)
  return

if __name__ == '__main__':
  main()