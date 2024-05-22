mkdir -p ./log/si
num_works=-1

if [ $# != 1 ]; then
    echo arg error
    exit 1
fi
evaluate=$1
# for data_name in 'lotka_volterra' 'hopf' 'cubic' 'vanderpol' 'rossler'
# do
# echo "------------------"
# echo "data:" $data_name
# echo "------------------"
logfn="./log/si/SI_eval_0519.txt"
nohup poetry run python _SI.py --num_works $num_works --evaluate $evaluate > $logfn &
# done