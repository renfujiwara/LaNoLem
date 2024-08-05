mkdir -p ./log/latent
num_works=-1
# num_works=1
today=$(date "+%Y%m%d")
for dataset_name in 'outdoor' 'ship'
do
logfn="./log/latent/Latent_${dataset_name}_eval_${today}.txt"
nohup poetry run python _Latent.py --dataset_name $dataset_name --num_works $num_works > $logfn &
done