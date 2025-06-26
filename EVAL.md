# Evaluation

## Running Evaluation

To evaluate EntitySAM on the VIPSeg dataset, follow these steps:

```bash
cd entitysam
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -u eval/eval_vipseg_entity_seg.py --ckpt_dir checkpoints/vit-l/
python -u eval/eval_vipseg_entity_seg.py --ckpt_dir checkpoints/vit-s/ --model_cfg configs/sam2.1_hiera_s.yaml --mask_decoder_depth 4
```

## Computing Metrics

After running the evaluation, compute the metrics using the following commands:

### Video Entity Quality (VEQ) Metric
```bash
python -u eval/metric/eval_veq.py -i checkpoints/vit-l/inference
```

### Segmentation and Tracking Quality (STQ) Metric
```bash
python -u eval/metric/eval_stq_vspw_clsag.py -i checkpoints/vit-l/inference
```

## Evaluation Results

The evaluation will generate results in the `checkpoints/vit-l/inference` directory. The metrics include:
- **VEQ**: Video Entity Quality score for class-agnostic entity segmentation performance
- **STQ**: Segmentation and Tracking Quality score for class-agnostic tracking consistency 