# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import motmetrics as mm
import motmetrics_custom as mm_custom
import numpy as np
import time
import argparse
import logging
from collections import OrderedDict
import json
import yaml
from cp_tracker import PubTracker as Tracker
import load_novatel_data, convert_novatel_to_pose
from scipy.spatial.transform import Rotation as R
import load_calibration
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

class_list = ['Car', 'Truck', 'Bus', 'Pedestrian']
# class_list = ['Pedestrian']

def parse_args():
    parser = argparse.ArgumentParser(description="""
        Compute metrics for trackers using CADC ground-truth data with data preprocess.
        """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_path', type=str, default='data/cadcd/', help='Path to CADC dataset')
    parser.add_argument('--out_dir', type=str, default='out/', help='Directory to save evaluation results')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='Confidence threshold for detections')
    return parser.parse_args()


def plot_bev(detections, T_global_gps, T_gps_lidar, dist_limit, frame, lidar_path, annotations_path, save_path):
    # Get lidar points
    scan_data = np.fromfile(lidar_path, dtype= np.float32) #numpy from file reads binary file
    lidar = scan_data.reshape((-1, 4))
    lidar_x = lidar[:,0]
    lidar_y = lidar[:,1]
    lidar_z = lidar [:,2]
    trunc_bool = (lidar_x < dist_limit) & (lidar_x > -dist_limit) & (lidar_y < dist_limit) & (lidar_y > -dist_limit)
    lidar_x = lidar_x[trunc_bool]
    lidar_y = lidar_y[trunc_bool]
    lidar_z = lidar_z[trunc_bool]
    lidar_x_plt = -1*lidar_y #in the image plot, the negative lidar y axis is the img x axis
    lidar_y_plt = lidar_x #the lidar x axis is the img y axis
    lidar_z_plt = lidar_z

    # Get ground truth boxes
    annotations_data = None
    with open(annotations_path) as f:
        annotations_data = json.load(f)
    T_Lidar_Cuboids = np.eye(4) * np.ones((len(annotations_data[frame]['cuboids']), 4, 4))
    lwh_cuboids = np.ones((len(annotations_data[frame]['cuboids']), 3))
    for i, cuboid in enumerate(annotations_data[frame]['cuboids']):
        T_Lidar_Cuboids[i,0:3,0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix()
        T_Lidar_Cuboids[i,0,3] = cuboid['position']['x']
        T_Lidar_Cuboids[i,1,3] = cuboid['position']['y']
        T_Lidar_Cuboids[i,2,3] = cuboid['position']['z']
        lwh_cuboids[i,0] = cuboid['dimensions']['y']
        lwh_cuboids[i,1] = cuboid['dimensions']['x']
        lwh_cuboids[i,2] = cuboid['dimensions']['z']
    trunc_bool = (T_Lidar_Cuboids[:,0,3] < dist_limit) & (T_Lidar_Cuboids[:,0,3] > -dist_limit) & (T_Lidar_Cuboids[:,1,3] < dist_limit) & (T_Lidar_Cuboids[:,1,3] > -dist_limit)
    T_Lidar_Cuboids = T_Lidar_Cuboids[trunc_bool]
    lwh_cuboids = lwh_cuboids[trunc_bool]
    frt = np.eye(4) * np.ones((len(T_Lidar_Cuboids), 4, 4))
    flt = np.eye(4) * np.ones((len(T_Lidar_Cuboids), 4, 4))
    brt = np.eye(4) * np.ones((len(T_Lidar_Cuboids), 4, 4))
    blt = np.eye(4) * np.ones((len(T_Lidar_Cuboids), 4, 4))
    frt[:,0:3,3] = np.array([lwh_cuboids[:,0]/2, lwh_cuboids[:,1]/2, lwh_cuboids[:,2]/2]).T
    flt[:,0:3,3] = np.array([lwh_cuboids[:,0]/2, -lwh_cuboids[:,1]/2, lwh_cuboids[:,2]/2]).T
    brt[:,0:3,3] = np.array([-lwh_cuboids[:,0]/2, lwh_cuboids[:,1]/2, lwh_cuboids[:,2]/2]).T
    blt[:,0:3,3] = np.array([-lwh_cuboids[:,0]/2, -lwh_cuboids[:,1]/2, lwh_cuboids[:,2]/2]).T
    frt = T_Lidar_Cuboids @ frt
    flt = T_Lidar_Cuboids @ flt
    brt = T_Lidar_Cuboids @ brt
    blt = T_Lidar_Cuboids @ blt
    boxes_x = np.array([frt[:,0,3], flt[:,0,3], blt[:,0,3], brt[:,0,3], frt[:,0,3]])
    boxes_y = np.array([frt[:,1,3], flt[:,1,3], blt[:,1,3], brt[:,1,3], frt[:,1,3]])
    boxes_x_plt = -1*boxes_y
    boxes_y_plt = boxes_x

    # Get predictions
    # pred_boxes_pos = np.array([detection['translation'] for detection in detections])
    # pred_boxes_yaw = np.array([Quaternion(detection['rotation']).yaw_pitch_roll[0] for detection in detections])
    pred_conf = [str(detection['detection_score'])[:4] for detection in detections]
    # T_lidar_cuboids = np.eye(4) * np.ones((len(pred_boxes_pos), 4, 4))
    # # T_lidar_cuboids[:,0:3,0:3] = R.from_euler('z', pred_boxes_yaw, degrees=False).as_matrix()
    # T_lidar_cuboids[:,0:3,3] = pred_boxes_pos
    # T_global_cuboids = T_global_gps @ T_gps_lidar @ T_lidar_cuboids
    # pred_x = T_global_cuboids[:,0,3]
    # pred_y = T_global_cuboids[:,1,3]
    pred_x_check = np.array([detection['ego_translation'][0] for detection in detections])
    pred_y_check = np.array([detection['ego_translation'][1] for detection in detections])
    # print(pred_x_check[0], pred_y_check[0], pred_x[0], pred_y[0])
    pred_x_plt = -1*pred_y_check
    pred_y_plt = pred_x_check
    # b = np.array([pred_x_check[0], pred_y_check[0]])
    # a = pred_boxes_pos[0,0:2]
    # angle = np.arctan2(b[1]-a[1], b[0]-a[0])
    # angle2 = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    
    # Plot
    cmap = "jet"    # Color map to use
    dpi = 100       # Image resolution
    fig, ax = plt.subplots(figsize=(2000/dpi, 2000/dpi), dpi=dpi)
    ax.plot(boxes_x_plt, boxes_y_plt, 'r')
    ax.scatter(lidar_x_plt, lidar_y_plt, s=1, c=lidar_z_plt, alpha=1.0, cmap=cmap, label='pc')
    ax.scatter(pred_x_plt, pred_y_plt, s=100, c='m', marker='*', alpha=1.0, label='det')
    for i, txt in enumerate(pred_conf):
        ax.annotate(txt, (pred_x_plt[i], pred_y_plt[i]))
    ax.axis('scaled')  # {equal, scaled}
    plt.xlim([-dist_limit, dist_limit])
    plt.ylim([-dist_limit, dist_limit])  
    ax.legend()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close()


def viz(gt_dir, det_file, res_dir, seqmap_path, viz_dir, data_path):
    # Get data
    with open(seqmap_path, 'r') as f:
        seqs = f.readlines()
        seqs = [seq.strip() for seq in seqs]
    gtfiles = [os.path.join(gt_dir, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(res_dir, '%s.txt' % i) for i in seqs]
    gt = OrderedDict([(seqs[i], (mm.io.loadtxt(f), os.path.join(gt_dir, seqs[i], 'seqinfo.ini'))) for i, f in enumerate(gtfiles)])
    ts = OrderedDict([(seqs[i], mm.io.loadtxt(f)) for i, f in enumerate(tsfiles)])
    with open(det_file, 'rb') as f:
        detections=json.load(f)['results']
    for seq, tracks in ts.items():
        if seq in gt:
            logging.info('Plotting %s...', seq)
            date_seq = seq[:-5] + '/' + seq[-4:]
            novatel_path = osp.join(data_path, date_seq, 'labeled/novatel/data/')
            novatel = load_novatel_data.load_novatel_data(novatel_path)
            poses = convert_novatel_to_pose.convert_novatel_to_pose(novatel)
            calib_path = data_path + seq[:-5] + "/" + "calib/"
            calib = load_calibration.load_calibration(calib_path)
            T_gps_lidar = np.linalg.inv(calib['extrinsics']['T_LIDAR_GPSIMU'])

            for frame in tracks.index.get_level_values(0).unique():
                # Plot gt and predictions in global frame
                dets_frame = detections[f'{date_seq}/{frame-1:010d}']
                det_x = []
                det_y = []
                vel_x = []
                vel_y = []
                dets_frame_new = []
                for detection in dets_frame:
                    if (detection['detection_name'] not in class_list) or (detection['detection_score'] < 0.5):
                        continue
                    T_lidar_box = np.eye(4)
                    T_lidar_box[0][3] = detection['ego_translation'][0]
                    T_lidar_box[1][3] = detection['ego_translation'][1]
                    T_lidar_box[2][3] = detection['ego_translation'][2]
                    T_global_gps = np.asarray(poses[frame-1])
                    T_global_box = T_global_gps @ T_gps_lidar @ T_lidar_box
                    det_x.append(T_global_box[0][3])
                    det_y.append(T_global_box[1][3])
                    vel_x.append(detection['velocity'][0])
                    vel_y.append(detection['velocity'][1])
                    # det_x.append(detection['translation'][0])
                    # det_y.append(detection['translation'][1])
                    dets_frame_new.append(detection)
                ax = gt[seq][0].loc[frame].plot.scatter(x='X', y='Y', s=20, c='r', marker='o',  label='gt')
                if frame+1 in tracks.index.get_level_values(0).unique():
                    for id in gt[seq][0].loc[frame].index:
                        if id in gt[seq][0].loc[frame+1].index:
                            ax.arrow(gt[seq][0].loc[frame].loc[id, 'X'], gt[seq][0].loc[frame].loc[id, 'Y'], gt[seq][0].loc[frame+1].loc[id, 'X']-gt[seq][0].loc[frame].loc[id, 'X'], gt[seq][0].loc[frame+1].loc[id, 'Y']-gt[seq][0].loc[frame].loc[id, 'Y'], head_width=2.0, head_length=2.0, fc='m', ec='m')
                # ax.scatter(det_x, det_y, s=10, c='g', label='detect')
                for i in range(len(det_x)):
                    ax.arrow(det_x[i], det_y[i], 0.3*vel_x[i], 0.3*vel_y[i], head_width=1.5, head_length=1.5, fc='k', ec='k')
                tracks.loc[frame].plot.scatter(x='X', y ='Y', s=4, c='blue', marker='*', label='track', ax=ax)
                ax.legend()
                viz_path = osp.join(viz_dir, f'{seq}_global_{frame:03d}.png')
                plt.savefig(viz_path)
                plt.close()

                # Plot in lidar frame: pointcloud, ground truth boxes, predictions
                viz_path = osp.join(viz_dir, f'{seq}_lidar_{frame:03d}.png')
                lidar_path = f"{data_path}/{date_seq}/labeled/lidar_points/data/{frame-1:010d}.bin"
                annotations_file = f"{data_path}/{date_seq}/3d_ann.json"
                bev_lim = 50
                T_global_gps = np.linalg.inv(np.asarray(poses[frame-1]))
                plot_bev(dets_frame_new, T_global_gps, T_gps_lidar, bev_lim, frame-1, lidar_path, annotations_file, viz_path)

                # TODO: make video?
            breakpoint()
        else:
            logging.warning('No ground truth for %s, skipping.', seq)

def evaluate(gt_dir, res_dir, seqmap_path, out_dir, dist='iou', distfields=None, distth=0.5):
    # Check predictions and ground truth are the same sequences
    with open(os.path.join(out_dir, 'seqmap.txt'), 'r') as f:
        seqs_track = set(f.readlines())
    with open(seqmap_path, 'r') as f:
        seqs_gt = set(f.readlines())
    assert seqs_track == seqs_gt, 'seqs in gt and tracking results are different'

    # Get data
    with open(seqmap_path, 'r') as f:
        seqs = f.readlines()
        seqs = [seq.strip() for seq in seqs]
    gtfiles = [os.path.join(gt_dir, i, 'gt/gt.txt') for i in seqs]
    tsfiles = [os.path.join(res_dir, '%s.txt' % i) for i in seqs]
    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    logging.info(seqs)
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')
    gt = OrderedDict([(seqs[i], (mm.io.loadtxt(f), os.path.join(gt_dir, seqs[i], 'seqinfo.ini'))) for i, f in enumerate(gtfiles)])
    ts = OrderedDict([(seqs[i], mm.io.loadtxt(f)) for i, f in enumerate(tsfiles)])
    
    # Compute metrics
    mh = mm.metrics.create()
    start_time = time.time()
    accs = []
    analysis = []
    names = []
    for k, tsacc in ts.items():
        if k in gt:
            logging.info('Evaluating %s...', k)
            # fd = io.open(out_dir + '/eval' + k + '.log', 'w')
            fd = ''
            # acc, ana = mm_custom.CLEAR_MOT_M(gt[k][0], tsacc, gt[k][1], dist, distfields, distth, include_all=True, vflag=fd)
            acc, ana = mm.utils.CLEAR_MOT_M(gt[k][0], tsacc, gt[k][1], dist, distfields, distth, include_all=True, vflag=fd)
            accs.append(acc)
            analysis.append(ana)
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)
    logging.info('Evaluation time: %.2f seconds.', time.time()-start_time)
    logging.info('Computing metrics')
    summary = mh.compute_many(accs, anas=analysis, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    
    # Print and save metrics
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    eval_results = {
        mm.io.motchallenge_metric_names[k]: v['OVERALL']
        for k, v in summary.to_dict().items()
    }
    for k, v in eval_results.items():
        if isinstance(v, float):
            eval_results[k] = float(f'{(v):.3f}')
    os.makedirs(out_dir, exist_ok=True)
    yaml_file_path = osp.join(out_dir, 'results.yaml')
    with open(yaml_file_path, 'w') as f:
        yaml.dump(eval_results, f)
    logging.info(f'Save results to {yaml_file_path}')

def format_cadc_gt(gt_path, data_path):
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
    val_set = {
        '2018_03_06_0001',
        '2018_03_06_0008',
        '2018_03_06_0016',
        '2018_03_07_0004',
        '2019_02_27_0009',
        '2019_02_27_0028',
        '2019_02_27_0016',
        '2019_02_27_0033',
        '2019_02_27_0040',
        '2019_02_27_0043',
        '2019_02_27_0054',
        '2019_02_27_0060',
        '2019_02_27_0065',
        '2019_02_27_0076',
        }
    logging.info('Generating seqmap.txt')
    seqmap_path = osp.join(gt_path, 'seqmap_all.txt')
    os.makedirs(gt_path, exist_ok=True)
    with open(seqmap_path, 'w+') as f:
        for date, seqs in cadcd.items():
            for seq in seqs:
                f.write(f'{date}_{seq}\n')
    seqmap_path = osp.join(gt_path, 'seqmap_val.txt')
    with open(seqmap_path, 'w+') as f:
        for seq in val_set:
            f.write(f'{seq}\n')

    logging.info('Generating ground truth files gt.txt and seqinfo.ini')
    for date, seqs in cadcd.items():
        calib_path = data_path + date + "/" + "calib/"
        calib = load_calibration.load_calibration(calib_path)
        T_gps_lidar = np.linalg.inv(calib['extrinsics']['T_LIDAR_GPSIMU'])
        for seq in seqs:
            ann_file = osp.join(data_path, date, seq, '3d_ann.json')
            with open(ann_file, 'r') as f:
                anns = json.load(f)
            novatel_path = osp.join(data_path, date, seq, 'labeled/novatel/data/')
            novatel = load_novatel_data.load_novatel_data(novatel_path)
            poses = convert_novatel_to_pose.convert_novatel_to_pose(novatel)
            gt_file = osp.join(gt_path, date + "_" + seq, 'gt', 'gt.txt')
            os.makedirs(osp.dirname(gt_file), exist_ok=True)
            with open(gt_file, 'w+') as f:
                frame_id = 0
                track_uuids = {}
                for ann in anns:
                    T_global_gps = poses[frame_id]
                    frame_id += 1
                    for cuboid in ann['cuboids']:
                        if cuboid['points_count'] > 0 and cuboid['label'] in class_list:
                            distance = np.linalg.norm(np.array([cuboid['position']['x'], cuboid['position']['y']]))
                            if distance > 50:
                                continue
                            T_lidar_cuboid = np.eye(4)
                            T_lidar_cuboid[0:3,0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix()
                            T_lidar_cuboid[0][3] = cuboid['position']['x']
                            T_lidar_cuboid[1][3] = cuboid['position']['y']
                            T_lidar_cuboid[2][3] = cuboid['position']['z']
                            T_global_cuboid = T_global_gps @ T_gps_lidar @ T_lidar_cuboid
                            x = T_global_cuboid[0,3]
                            y = T_global_cuboid[1,3]
                            track_uuid = cuboid['uuid']
                            if track_uuid not in track_uuids:
                                track_uuids[track_uuid] = len(track_uuids) + 1
                            track_id = track_uuids[track_uuid]
                            f.write(f'{frame_id},{track_id},{x},{y},0,0,1,1,1.0\n')
            seqinfo_file = osp.join(gt_path, date + "_" + seq, 'seqinfo.ini')
            with open(seqinfo_file, 'w') as f:
                f.write(f'[Sequence]\nname={date}_{seq}\nseqLength={frame_id}\n')
    
    logging.info('Done gt generation')


def run_tracking(det_file, res_dir, data_path, conf_thresh):
    max_age = 3
    hungarian = True
    tracker = Tracker(max_age, hungarian, conf_thresh)
    predictions_in_lidar = True

    # Get predictions and sequence tokens
    with open(det_file, 'rb') as f:
        predictions=json.load(f)['results']
    seq_tokens = {}
    for sample_token in predictions:
        seq_token = sample_token[:-11]
        seq_token = seq_token.replace('/', '_')
        if seq_token not in seq_tokens:
            seq_tokens[seq_token] = []
        seq_tokens[seq_token].append(sample_token)
    
    # Hack to fix tranformation issue
    if predictions_in_lidar:
        for sample_token in predictions:
            date = sample_token[:10]
            seq = sample_token[11:15]
            frame = int(sample_token[-3:])
            novatel_path = osp.join(data_path, date, seq, 'labeled/novatel/data/')
            novatel = load_novatel_data.load_novatel_data(novatel_path)
            poses = convert_novatel_to_pose.convert_novatel_to_pose(novatel)
            T_global_gps = np.asarray(poses[frame])
            calib_path = data_path + date + "/" + "calib/"
            calib = load_calibration.load_calibration(calib_path)
            T_gps_lidar = np.linalg.inv(calib['extrinsics']['T_LIDAR_GPSIMU'])
            T_global_lidar = T_global_gps @ T_gps_lidar
            for i, detection in enumerate(predictions[sample_token]):
                T_lidar_box = np.eye(4)
                T_lidar_box[0:3,0:3] = R.from_quat(detection['rotation']).as_matrix()
                T_lidar_box[0][3] = detection['translation'][0]
                T_lidar_box[1][3] = detection['translation'][1]
                T_lidar_box[2][3] = detection['translation'][2]
                T_global_box = T_global_lidar @ T_lidar_box
                predictions[sample_token][i]['translation'] = [T_global_box[0][3], T_global_box[1][3], T_global_box[2][3]]
                predictions[sample_token][i]['rotation'] = R.from_matrix(T_global_box[0:3,0:3]).as_quat().tolist()
                # T_lidar_vel = np.eye(4)
                # T_lidar_vel[0,3] = detection['velocity'][0]
                # T_lidar_vel[1,3] = detection['velocity'][1]
                # T_lidar_vel[2,3] = 0.0
                # T_global_vel = T_global_lidar @ T_lidar_vel
                # predictions[sample_token][i]['velocity'] = [T_global_vel[0][3], T_global_vel[1][3], T_global_vel[2][3]]
    
    # Run tracking
    time_lag = 0.3
    track_results = {}
    for seq in seq_tokens:
        logging.info(f'Running tracking on {seq}')
        tracker.reset()
        for sample_token in seq_tokens[seq]:
            outputs = tracker.step_centertrack(predictions[sample_token], time_lag)
            annos = []
            for item in outputs:
                if item['active'] == 0:
                    continue 
                nusc_anno = {
                    "sample_token": sample_token,
                    "translation": item['translation'],
                    "ego_translation": item['ego_translation'],
                    "size": item['size'],
                    "rotation": item['rotation'],
                    "velocity": item['velocity'],
                    "tracking_id": str(item['tracking_id']),
                    "tracking_name": item['detection_name'],
                    "tracking_score": item['detection_score'],
                }
                annos.append(nusc_anno)
            track_results.update({sample_token: annos})

    # Save tracking results
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(os.path.join(res_dir, 'tracking_result.json'), "w+") as f:
        json.dump(track_results, f)
    for seq in seq_tokens:
        with open(os.path.join(res_dir, f'{seq}.txt'), 'w+') as f:
            frame_id = 0
            for sample_token in seq_tokens[seq]:
                frame_id += 1
                for item in track_results[sample_token]:
                    f.write(f'{frame_id},{item["tracking_id"]},{item["translation"][0]},{item["translation"][1]},0,0,1,1,1.0\n')
    seqmap_path = os.path.join(res_dir, 'seqmap.txt')
    with open(seqmap_path, 'w+') as f:
        for seq in seq_tokens:
            f.write(f'{seq}\n')
    logging.info('Done tracking')

def unittest_eval():
    gt_dir = 'data/MOT17/gt_root/' 
    res_dir = 'data/MOT17/test_root/'
    seqmap_path = 'data/MOT17/seqmap.txt'
    evaluate(gt_dir, res_dir, seqmap_path, res_dir)

def main():
    # Get arguments
    args = parse_args()
    out_dir = osp.join(args.out_dir, 'mot', 'test')
    det_file = osp.join(args.out_dir, 'det', 'results_lidar_frame.json')
    gt_dir = osp.join(args.out_dir, 'mot', 'gt')
    viz_dir = osp.join(args.out_dir, 'mot', 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    seqmap_path = osp.join(gt_dir, 'seqmap_val.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')
    print('confidence threshold:', args.conf_thresh)

    # Generate ground truth files, run tracking, evaluate and visualize
    # format_cadc_gt(gt_dir, args.data_path)
    # run_tracking(det_file, out_dir, args.data_path, args.conf_thresh)
    # evaluate(gt_dir, out_dir, seqmap_path, out_dir, dist='l2', distfields=['X', 'Y'], distth=2.0)
    viz(gt_dir, det_file, out_dir, seqmap_path, viz_dir, args.data_path)


if __name__ == '__main__':
    main()
    # unittest_eval()

