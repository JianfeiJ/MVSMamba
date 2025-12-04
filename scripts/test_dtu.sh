export CUDA_VISIBLE_DEVICES=2

DTU_TESTPATH="/datasets/DTU/DTU_Dataset/dtu_test/"
DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE='checkpoints/dtu/finalmodel_9.ckpt' # dtu pretrained model


exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./outputs/dtu/"$exp
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi
DTU_OUT_DIR="./outputs/dtu"$exp
if [ ! -d $DTU_OUT_DIR ]; then
    mkdir -p $DTU_OUT_DIR
fi

python test_dtu_dypcd.py --dataset=general_eval4 --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --outdir $DTU_OUT_DIR\
            --ndepths 32,16,8,4 --depth_inter_r 2.0,1.0,1.0,0.5 --group_cor_dim 4,4,4,4 --conf 0.55 --use_raw_train --attn_temp 2 --num_view 5 --num_worker 2 $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt

