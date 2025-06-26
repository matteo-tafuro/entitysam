# -------------------------------------------------------------------
# Class Agnostic Video Panoptic Segmentation
#
# VEQ evaluation code by tube (video segment) matching
# Inference on every frames and evaluation on every 5 frames.
# ------------------------------------------------------------------
from tqdm import tqdm
from functools import partial

import argparse
import sys
import os
import os.path
import numpy as np
from PIL import Image
import multiprocessing as mp
import time
import json
from collections import defaultdict
import copy
import pdb

class PQStatCatAgnostic:
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self

class PQStatAgnostic:
    def __init__(self):
        self.pq_stat = PQStatCatAgnostic()

    def __iadd__(self, pq_stat):
        self.pq_stat += pq_stat.pq_stat
        return self

    def pq_average(self):
        pq, sq, rq, n = 0, 0, 0, 0
        iou = self.pq_stat.iou
        tp = self.pq_stat.tp
        fp = self.pq_stat.fp
        fn = self.pq_stat.fn
        if tp + fp + fn == 0:
            return {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'iou': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        n += 1
        pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
        sq_class = iou / tp if tp != 0 else 0
        rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
        return {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn}

def vpq_compute_single_core_agnostic(nframes, gt_pred_set):
    OFFSET = 256 * 256 * 256
    VOID = 0
    vpq_stat = PQStatAgnostic()

    for idx in range(0, len(gt_pred_set)-nframes+1): 
        vid_pan_gt, vid_pan_pred = [], []
        gt_segms_list, pred_segms_list = [], []

        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(gt_pred_set[idx:idx+nframes]):
            gt_pan = Image.open(gt_pan)
            pred_pan = Image.open(pred_pan)
            assert gt_pan.size == pred_pan.size, f"Dismatch shape {gt_pan.size} and {pred_pan.size}"
            gt_pan = np.array(gt_pan)
            pred_pan = np.array(pred_pan)

            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256
            gt_segms = {}
            for el in gt_json['segments_info']:
                if el['id'] in gt_segms:
                    gt_segms[el['id']]['area'] += copy.deepcopy(el['area'])
                else:
                    gt_segms[el['id']] = copy.deepcopy(el)
            pred_segms = {}
            for el in pred_json['segments_info']:
                if el['id'] in pred_segms:
                    pred_segms[el['id']]['area'] += copy.deepcopy(el['area'])
                else:
                    pred_segms[el['id']] = copy.deepcopy(el)
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('Segment with ID {} is presented in PNG and not presented in JSON.'.format(label))
                if 'area' in pred_segms[label]:
                    pred_area = pred_segms[label]['area']
                    assert pred_area == label_cnt, f'Mismatch numbers of {pred_area} and {label_cnt}'
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
            
            if len(pred_labels_set) != 0:
                raise KeyError('The following segment IDs {} are presented in JSON and not presented in PNG.'.format(list(pred_labels_set)))

            vid_pan_gt.append(pan_gt)
            vid_pan_pred.append(pan_pred)
            gt_segms_list.append(gt_segms)
            pred_segms_list.append(pred_segms)

        vid_pan_gt = np.stack(vid_pan_gt)
        vid_pan_pred = np.stack(vid_pan_pred)
        vid_gt_segms, vid_pred_segms = {}, {}
        for gt_segms, pred_segms in zip(gt_segms_list, pred_segms_list):
            for k in gt_segms.keys():
                if not k in vid_gt_segms:
                    vid_gt_segms[k] = gt_segms[k]
                else:
                    vid_gt_segms[k]['area'] += gt_segms[k]['area']
            for k in pred_segms.keys():
                if not k in vid_pred_segms:
                    vid_pred_segms[k] = pred_segms[k]
                else:
                    vid_pred_segms[k]['area'] += pred_segms[k]['area']

        vid_pan_gt_pred = vid_pan_gt.astype(np.uint64) * OFFSET + vid_pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(vid_pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        gt_matched = set()
        pred_matched = set()
        tp = 0
        fp = 0
        fn = 0

        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple

            if gt_label not in vid_gt_segms:
                continue
            if pred_label not in vid_pred_segms:
                continue
            if vid_gt_segms[gt_label]['iscrowd'] == 1:
                continue

            union = vid_pred_segms[pred_label]['area'] + vid_gt_segms[gt_label]['area'] - intersection 
            if intersection > (union - gt_pred_map.get((VOID, pred_label), 0)):
                print('pred and gt labels:', pred_label, gt_label)
                print('interseaction, pred_area, and gt_area:', intersection, vid_pred_segms[pred_label]['area'], vid_gt_segms[gt_label]['area'])
                print('union before and after bg pixel removal:', union, union - gt_pred_map.get((VOID, pred_label), 0))
                print(vid_gt_segms.keys())
            union = union - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            assert iou <= 1.0, f'INVALID IOU VALUE: {iou} on the gt_label {int(gt_label)} and the pred_label {int(pred_label)}'
            if iou > 0.5:
                vpq_stat.pq_stat.tp += 1
                vpq_stat.pq_stat.iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)
                tp += 1

        crowd_labels = set()
        for gt_label, gt_info in vid_gt_segms.items():
            if gt_label in gt_matched:
                continue
            if gt_info['iscrowd'] == 1:
                crowd_labels.add(gt_label)
                continue
            vpq_stat.pq_stat.fn += 1
            fn += 1

        for pred_label, pred_info in vid_pred_segms.items():
            if pred_label in pred_matched:
                continue
            intersection = gt_pred_map.get((VOID, pred_label), 0)

            if any((crowd_label, pred_label) in gt_pred_map for crowd_label in crowd_labels):
                intersection += sum(gt_pred_map.get((crowd_label, pred_label), 0) for crowd_label in crowd_labels)

            if intersection / pred_info['area'] > 0.5:
                continue
            vpq_stat.pq_stat.fp += 1
            fp += 1

    return vpq_stat

def vpq_compute_agnostic(gt_pred_split, nframes, output_dir):
    start_time = time.time()
    vpq_stat = PQStatAgnostic()
    for idx, gt_pred_set in enumerate(tqdm(gt_pred_split)):
        tmp = vpq_compute_single_core_agnostic(gt_pred_set=gt_pred_set, nframes=nframes)
        vpq_stat += tmp

    k = (nframes-1)*5
    print('==> %d-frame vpq_stat:' % (k), time.time() - start_time, 'sec')
    results = vpq_stat.pq_average()

    vpq_all = 100 * results['pq']
    vpq_sq = 100 * results['sq']
    vpq_rq = 100 * results['rq']

    save_name = os.path.join(output_dir, 'vpq-agnostic-%d.txt' % (k))
    f = open(save_name, 'w') if save_name else None
    f.write("================================================\n")
    f.write("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N\n"))
    f.write("-" * (10 + 7 * 4) + '\n')
    f.write("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}\n".format("All", 100 * results['pq'], 100 * results['sq'], 100 * results['rq'], 1))
    f.write("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}\n".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"))
    f.write("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}\n".format(0, 100 * results['pq'], 100 * results['sq'], 100 * results['rq'], results['iou'], results['tp'], results['fp'], results['fn']))
    if save_name:
        f.close()

    return vpq_all, vpq_sq, vpq_rq

def parse_args():
    parser = argparse.ArgumentParser(description='VPSNet eval')

    parser.add_argument('--submit_dir', '-i',
                        type=str,
                        help='test output directory', required=True)

    parser.add_argument('--truth_dir', type=str,
                        default="datasets/VIPSeg_720P/panomasksRGB/",
                        help='ground truth directory. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panomasksRGB '
                             'after running the conversion script')

    parser.add_argument('--pan_gt_json_file', type=str,
                        default="datasets/VIPSeg_720P/panoptic_gt_VIPSeg_val.json",
                        help='ground truth JSON file. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panoptic_gt_'
                             'VIPSeg_val.json after running the conversion script')

    parser.add_argument("--num_processes", type=int, default=8)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    _root = '/'.join(os.getenv("DETECTRON2_DATASETS", "datasets").split('/')[:-1])
    submit_dir = os.path.join(_root, args.submit_dir)
    truth_dir =  os.path.join(_root, args.truth_dir)
    output_dir = submit_dir
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    start_all = time.time()
    pan_pred_json_file = os.path.join(submit_dir, 'pred.json')
    with open(pan_pred_json_file, 'r') as f:
        pred_jsons = json.load(f)
    pan_gt_json_file = os.path.join(_root, args.pan_gt_json_file)
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)

    categories = gt_jsons['categories']
    categories = {el['id']: el for el in categories}

    start_time = time.time()

    pred_annos = pred_jsons['annotations']
    pred_j = {}
    for p_a in pred_annos:
        pred_j[p_a['video_id']] = p_a['annotations']
    gt_annos = gt_jsons['annotations']
    gt_j = {}
    for g_a in gt_annos:
        gt_j[g_a['video_id']] = g_a['annotations']

    gt_pred_split = []

    pbar = tqdm(gt_jsons['videos'])
    for video_images in pbar:
        pbar.set_description(video_images['video_id'])
    
        video_id = video_images['video_id']
        gt_image_jsons = video_images['images']
        if video_id not in pred_j:
            print(f"{video_id} does not in prediced json, please double check!!")
            continue
        gt_js = gt_j[video_id]
        pred_js = pred_j[video_id]
        assert len(gt_js) == len(pred_js)
        gt_pans = []
        pred_pans = []
        for imgname_j in gt_image_jsons:
            imgname = imgname_j['file_name']

            pred_pans.append(os.path.join(submit_dir, 'pan_pred', video_id, imgname))
            gt_pans.append(os.path.join(truth_dir, video_id, imgname))

        gt_pred_split.append(list(zip(gt_js, pred_js, gt_pans, pred_pans, gt_image_jsons)))

    start_time = time.time()
    vpq_all, vpq_sq, vpq_rq = [], [], []

    for nframes in [1, 2, 4, 6, 8]:
        gt_pred_split_ = copy.deepcopy(gt_pred_split)
        vpq_all_, vpq_sq_, vpq_rq_ = vpq_compute_agnostic(
            gt_pred_split_, nframes, output_dir)
        del gt_pred_split_
        print(vpq_all_, vpq_sq_, vpq_rq_)
        vpq_all.append(vpq_all_)
        vpq_sq.append(vpq_sq_)
        vpq_rq.append(vpq_rq_)

    output_filename = os.path.join(output_dir, 'vpq-final-agnostic.txt')
    output_file = open(output_filename, 'w')
    output_file.write("vpq_all:%.4f\n" % (sum(vpq_all) / len(vpq_all)))
    output_file.write("vpq_sq:%.4f\n" % (sum(vpq_sq) / len(vpq_sq)))
    output_file.write("vpq_rq:%.4f\n" % (sum(vpq_rq) / len(vpq_rq)))
    output_file.close()
    print("vpq_all:%.4f\n" % (sum(vpq_all) / len(vpq_all)))
    print("vpq_sq:%.4f\n" % (sum(vpq_sq) / len(vpq_sq)))
    print("vpq_rq:%.4f\n" % (sum(vpq_rq) / len(vpq_rq)))
    print('==> All:', time.time() - start_all, 'sec')


if __name__ == "__main__":
    main()
