# -------------------------------------------------------------------
# Class Agnostic Video Panoptic Segmentation Evaluation
# ------------------------------------------------------------------

import argparse
import sys
import os
import numpy as np
from PIL import Image
import time
import json
from tqdm import tqdm
import segmentation_and_tracking_quality_clsag as numpy_stq

def parse_args():
    parser = argparse.ArgumentParser(description='Class Agnostic VSPW eval')
    parser.add_argument('--submit_dir', '-i', type=str,
                        help='test output directory')

    parser.add_argument('--truth_dir', type=str, default="datasets/VIPSeg_720P/panomasksRGB/",
                        help='ground truth directory. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panomasksRGB '
                             'after running the conversion script')

    parser.add_argument('--pan_gt_json_file', type=str, default="datasets/vipseg_dvis_ver/VIPSeg_720P/panoptic_gt_VIPSeg_val.json",
                        help='ground truth JSON file. Point this to <BASE_DIR>/VIPSeg/VIPSeg_720P/panoptic_gt_'
                             'VIPSeg_val.json after running the conversion script')

    args = parser.parse_args()
    return args

def convert_to_class_agnostic(json_data):
    """Convert all category IDs to single entity class."""
    entity_class_id = 1
    
    # Make a deep copy to avoid modifying original data
    json_data = json_data.copy()
    
    # Convert all categories to single entity class
    json_data['categories'] = [{
        'id': entity_class_id,
        'name': 'entity',
        'isthing': 1  # Everything is a "thing" to be tracked
    }]
    
    # Update all annotations to use single class
    for video_ann in json_data['annotations']:
        for frame_ann in video_ann['annotations']:
            for segment in frame_ann['segments_info']:
                segment['category_id'] = entity_class_id
                
    return json_data

# Add this function near the top of the file
def save_results(output_dir, result, args):
    """Save evaluation results to a txt file."""
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    result_file = os.path.join(output_dir, f'eval_results_{timestamp}.txt')
    
    with open(result_file, 'w') as f:
        # Write evaluation settings
        f.write('='*50 + '\n')
        f.write('Evaluation Settings:\n')
        f.write(f'Submit Directory: {args.submit_dir}\n')
        f.write(f'Ground Truth Directory: {args.truth_dir}\n')
        f.write(f'GT JSON: {args.pan_gt_json_file}\n')
        f.write('='*50 + '\n\n')
        
        # Write main results
        f.write('Main Results:\n')
        f.write('-'*50 + '\n')
        f.write(f'STQ : {result["STQ"]:.4f}\n')
        f.write(f'AQ : {result["AQ"]:.4f}\n')
        f.write(f'IoU: {result["IoU"]:.4f}\n\n')
        
        # Write per-sequence results
        f.write('Per-Sequence Results:\n')
        f.write('-'*50 + '\n')
        f.write('STQ per sequence:\n')
        for id_, stq in zip(result['ID_per_seq'], result['STQ_per_seq']):
            f.write(f'  {id_}: {stq:.4f}\n')
        
        f.write('\nAQ per sequence:\n')
        for id_, aq in zip(result['ID_per_seq'], result['AQ_per_seq']):
            f.write(f'  {id_}: {aq:.4f}\n')
            
        f.write('\nSequence Lengths:\n')
        for id_, length in zip(result['ID_per_seq'], result['Length_per_seq']):
            f.write(f'  {id_}: {length}\n')
    
    print(f'\nResults saved to: {result_file}')
    return result_file

def main():
    # Modified to use single class
    n_classes = 2  # Changed from 124 to 2
    ignore_label = 255
    bit_shit = 16
    entity_class_id = 1  # We'll use this as our single class ID
    
    args = parse_args()
    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    output_dir = submit_dir
    
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    start_all = time.time()
    
    # Load and convert JSONs to class-agnostic format
    pan_pred_json_file = os.path.join(submit_dir, 'pred.json')
    with open(pan_pred_json_file, 'r') as f:
        pred_jsons = json.load(f)
    pred_jsons = convert_to_class_agnostic(pred_jsons)
    
    pan_gt_json_file = args.pan_gt_json_file
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)
    gt_jsons = convert_to_class_agnostic(gt_jsons)

    # Modified to treat everything as "thing" with single class
    thing_list_ = [entity_class_id]
    
    stq_metric = numpy_stq.STQuality(n_classes, thing_list_, ignore_label,
                                     bit_shit, 2**24)

    pred_annos = pred_jsons['annotations']
    pred_j = {}
    for p_a in pred_annos:
        pred_j[p_a['video_id']] = p_a['annotations']
    gt_annos = gt_jsons['annotations']
    gt_j = {}
    for g_a in gt_annos:
        gt_j[g_a['video_id']] = g_a['annotations']

    pbar = tqdm(gt_jsons['videos'])
    for seq_id, video_images in enumerate(pbar):
        video_id = video_images['video_id']
        pbar.set_description(video_id)

        gt_image_jsons = video_images['images']
        gt_js = gt_j[video_id]
        pred_js = pred_j[video_id]
        assert len(gt_js) == len(pred_js)

        gt_pans = []
        pred_pans = []
        for imgname_j in gt_image_jsons:
            imgname = imgname_j['file_name']
            image = np.array(Image.open(os.path.join(submit_dir, 'pan_pred', video_id, imgname)))
            pred_pans.append(image)
            image = np.array(Image.open(os.path.join(truth_dir, video_id, imgname)))
            gt_pans.append(image)

        # Create instance number mapping
        gt_id_to_ins_num_dic = {}
        list_tmp = []
        for segm in gt_js:
            for img_info in segm['segments_info']:
                id_tmp_ = img_info['id']
                if id_tmp_ not in list_tmp:
                    list_tmp.append(id_tmp_)
        for ii, id_tmp_ in enumerate(list_tmp):
            gt_id_to_ins_num_dic[id_tmp_] = ii
            
        pred_id_to_ins_num_dic = {}
        list_tmp = []
        for segm in pred_js:
            for img_info in segm['segments_info']:
                id_tmp_ = img_info['id']
                if id_tmp_ not in list_tmp:
                    list_tmp.append(id_tmp_)
        for ii, id_tmp_ in enumerate(list_tmp):
            pred_id_to_ins_num_dic[id_tmp_] = ii

        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(list(zip(gt_js, pred_js, gt_pans, pred_pans, gt_image_jsons))):
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

            ground_truth_instance = np.ones_like(pan_gt) * 255
            ground_truth_semantic = np.ones_like(pan_gt) * 255
            for el in gt_json['segments_info']:
                id_ = el['id']
                # All objects are the same class
                ground_truth_semantic[pan_gt == id_] = entity_class_id
                ground_truth_instance[pan_gt == id_] = gt_id_to_ins_num_dic[id_]

            ground_truth = ((ground_truth_semantic << bit_shit) + ground_truth_instance)

            prediction_instance = np.ones_like(pan_pred) * 255
            prediction_semantic = np.ones_like(pan_pred) * 255

            for el in pred_json['segments_info']:
                id_ = el['id']
                # All objects are the same class
                prediction_semantic[pan_pred == id_] = entity_class_id
                prediction_instance[pan_pred == id_] = pred_id_to_ins_num_dic[id_]
            
            prediction = ((prediction_semantic << bit_shit) + prediction_instance)  

            stq_metric.update_state(ground_truth.astype(dtype=np.int32),
                              prediction.astype(dtype=np.int32), seq_id) 
    result = stq_metric.result()         
    print('*'*100)
    print('STQ : {}'.format(result['STQ']))
    print('AQ :{}'.format(result['AQ']))
    print('IoU:{}'.format(result['IoU']))
    print('STQ_per_seq')
    print(result['STQ_per_seq'])
    print('AQ_per_seq')
    print(result['AQ_per_seq'])
    print('ID_per_seq')
    print(result['ID_per_seq'])
    print('Length_per_seq')
    print(result['Length_per_seq'])
    print('*'*100)
    # Save results to file
    result_file = save_results(output_dir, result, args)



if __name__ == "__main__":
    main()