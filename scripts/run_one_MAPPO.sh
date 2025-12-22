for i in 0.001
do
python main_MAPPO.py -epochs 1000 -runs 1 \
    -L_num 200 -alpha 1e-3 -gamma 0.99 \
    -clip_epsilon 0.2 -question 2 -ppo_epochs 1 \
    -batch_size 1 -gae_lambda 0.95 \
    -delta 0.5 -rho 0.001 -seed 666
done
