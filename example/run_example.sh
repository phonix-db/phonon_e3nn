
file_data=../DATA/data_all.csv

num_data=100
seed=20
outdir=./out_N${num_data}/seed${seed}

## for test job
num_epochs=10
## for accurate calculation
#num_epochs=1000

python ../tools/run_prediction.py \
    --file_data $file_data \
    --num_data  $num_data \
    --outdir    $outdir \
    --target    kcumu_norm_mfp \
    --seed      $seed \
    --r_max     4.3 \
    --num_epochs       $num_epochs \
    --num_epochs_limit 1000 \
    --patience         50 \
    --nprocs   4 \
    --batch_size   4 \
    --lr           0.05000000 \
    --lr_min       0.01500000 \
    --gamma        0.95 \
    --weight_decay 0.00 \
    --optimizer    adam \
    --random_split 1

