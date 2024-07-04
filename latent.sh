mkdir -p ./log/latent
num_works=-1
# num_works=1
if [ $# != 1 ]; then
    echo arg error
    exit 1
fi
dataset_name=$1
logfn="./log/latent/Latent_eval.txt"
nohup poetry run python _Latent.py --dataset_name $dataset_name --num_works $num_works > $logfn &