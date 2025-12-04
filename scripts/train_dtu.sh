export CUDA_VISIBLE_DEVICES=1,2
DTU_TRAINING="/datasets/DTU/DTU_Dataset/dtu_training/"
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu_raw"$exp
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

# high_res training 1024x1280
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1511 train_dtu.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=2 --trainpath=$DTU_TRAINING --summary_freq 100 \
         --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 4,4,4,4 --pin_m --rt --attn_temp 2 --use_raw_train --epochs 10 --lr=0.001 --lrepochs 6,8,9:2 --lr_scheduler 'MS' \
         --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt

# low_res training 512x640
#python -m torch.distributed.launch --nproc_per_node=2 --master_port=1418 train_dtu.py --logdir $DTU_LOG_DIR --dataset=dtu_yao4 --batch_size=4 --trainpath=$DTU_TRAINING --summary_freq 100 \
#         --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 4,4,4,4 --pin_m --rt --attn_temp 2 --epochs 15 --lr=0.001 --lrepochs 10,12,14:2 --lr_scheduler 'MS' \
#         --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt