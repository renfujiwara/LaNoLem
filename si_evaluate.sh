mkdir -p ./log/si
num_works=-1

if [ $# != 1 ]; then
    echo arg error
    exit 1
fi

today=$(date "+%Y%m%d")

evaluate=$1
# for data_name in 'lotka_volterra' 'hopf' 'cubic' 'vanderpol' 'rossler'
# do
# echo "------------------"
# echo "data:" $data_name
# echo "------------------"
logfn="./log/si/SI_eval_${today}.txt"
nohup poetry run python _SI.py --num_works $num_works --date $today --evaluate $evaluate > $logfn &
# done