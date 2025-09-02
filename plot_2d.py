center_crop = True
data_path = "/home/trail/workspace/cadc_devkit/data/cadcd/"
viz_path = "/home/trail/workspace/cadc_devkit/viz/"
img_w = 1280
img_h = 1024
if center_crop:
  cam_list = [0, 4]
else:
  cam_list = [0, 1, 2, 3, 4, 5, 6, 7]
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
for cam in cam_list:
    cam = str(cam)
    viz_filepaths = []
    for date in cadcd:
        for seq in cadcd[date]:
            timestamps = data_path + date + "/" + seq + "/labeled/image_0" + cam + "/timestamps.txt"
            with open(timestamps) as f:
                num_timestamps = sum(1 for line in f)
            for frame in range(num_timestamps):
                img_path = data_path + date + "/" + seq + "/labeled/image_0" + cam + "/data/" + format(frame, '010') + ".png"
                img = cv2.imread(img_path)
                if center_crop:
                    img = img[int(img_h/4):int(3*img_h/4), int(img_w/4):int(3*img_w/4), :]
                # TODO: load detections and convert to rect_list format ((x1,y1), (x2,y2))
                for i in range(len(rect_list)):
                    cv2.rectangle(img, rect_list[i][0], rect_list[i][1], [0, 255, 0], thickness=1, lineType=8, shift=0)
                if center_crop:
                    viz_filepath = viz_path + "cam" + cam + "_crop/" + date + "_" + seq + "_" + str(frame) + ".png"
                else:
                    viz_filepath = viz_path + "cam" + cam + "/" + date + "_" + seq + "_" + str(frame) + ".png"
                os.makedirs(os.path.dirname(viz_filepath), exist_ok=True)
                cv2.imwrite(viz_filepath, img)
