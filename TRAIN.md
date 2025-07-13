# Train

## Dataset preparation

Prepare the [coco](https://cocodataset.org/) dataset in `./datasets/coco`

```
datasets/
└── coco/
    ├── train2017/
    ├── annotations/
    └── panoptic_train2017/
```


## Training

Example script for training EntitySAM with ViT-S backbone

```bash
export BASE_OUTPUT_DIR="output/vits"

# Run the Python training script with distributed GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
      --num-gpus 4 \
      --dist-url tcp://127.0.0.1:50159 \
      --resume \
      OUTPUT_DIR "${BASE_OUTPUT_DIR}_stage1" \
      SOLVER.IMS_PER_BATCH 4 \
      SOLVER.MAX_ITER 81000 \
      SOLVER.STEPS 60000, \
      SOLVER.CHECKPOINT_PERIOD 10000 \
      INPUT.SAMPLING_FRAME_NUM 1 \
      MODEL.MASK_DECODER_DEPTH 4 \
      MODEL.NAME "vits" \

echo "Stage 1 Training script executed."

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py \
      --num-gpus 4 \
      --dist-url tcp://127.0.0.1:50159 \
      --resume \
      OUTPUT_DIR "${BASE_OUTPUT_DIR}_stage2"\
      SOLVER.IMS_PER_BATCH 4 \
      SOLVER.MAX_ITER 10100 \
      MODEL.MASK_DECODER_DEPTH 4 \
      MODEL.WEIGHTS "${BASE_OUTPUT_DIR}_stage1/model_0079999.pth" \
      MODEL.NAME "vits" \

```

## Evaluation
See [Eval instruction](./EVAL.md)