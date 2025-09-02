
from datetime import datetime

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
    data_time_diffs = []
    for cam in range(8):
        cam = str(cam)
        cam_time_diffs = []
        for date in cadcd:
            date_time_diffs = []
            for seq in cadcd[date]:
                timestamps_path = base_path + date + "/" + seq + "/labeled/image_0" + cam + "/timestamps.txt"
                with open(timestamps_path) as file:
                    seq_time_diffs = []
                    timestamps = [line.rstrip() for line in file]
                    for i in range(len(timestamps)):
                        if i == 0:
                            continue                        
                        timestamp_0 = datetime.strptime(timestamps[i-1][:-3], "%Y-%m-%d %H:%M:%S.%f")
                        timestamp_1 = datetime.strptime(timestamps[i][:-3], "%Y-%m-%d %H:%M:%S.%f")
                        timestamp_diff = timestamp_1 - timestamp_0
                        seq_time_diffs.append(timestamp_diff.total_seconds())
                assert len(set(seq_time_diffs)) == 1
                date_time_diffs.append(seq_time_diffs[0])
            assert len(set(date_time_diffs)) == 1
        cam_time_diffs.append(date_time_diffs[0])
        assert len(set(cam_time_diffs)) == 1
    data_time_diffs.append(cam_time_diffs[0])
    assert len(set(data_time_diffs)) == 1
    print("All time differences are equal with a value of", data_time_diffs[0])


if __name__ == "__main__":
    main()