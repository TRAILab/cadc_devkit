
from datetime import datetime
import cv2

cadcd = {
    '2018_03_06': [
        '0001','0002','0005','0006','0008','0009','0010',
        '0012','0013','0015','0016','0018'
    ],
    '2018_03_07': [
        '0001','0002','0004','0005','0006','0007'
    ],
    '2019_02_27': [
        '0002','0003','0004','0005','0006','0008','0009','0010',
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

def main():
    base_path = "/home/trail/workspace/cadc_devkit/data/cadcd/"
    img_w_ = 1280
    img_h_ = 1024
    for cam in range(8):
        cam = str(cam)
        for date in cadcd:
            for seq in cadcd[date]:
                timestamps = base_path + date + "/" + seq + "/labeled/image_0" + cam + "/timestamps.txt"
                with open(timestamps) as f:
                    num_timestamps = sum(1 for line in f)
                for frame in range(num_timestamps):
                    img_path = base_path + date + "/" + seq + "/labeled/image_0" + cam + "/data/" + format(frame, '010') + ".png"
                    img = cv2.imread(img_path)
                    img_h, img_w = img.shape[:2]
                    assert img_h == img_h_ and img_w == img_w_, "Image dimensions do not match"
    print("All images have the correct dimensions")


if __name__ == "__main__":
    main()