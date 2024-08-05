num_works=-1

# if [ $# != 1 ]; then
#     echo arg error
#     exit 1
# fi

today=$(date "+%Y%m%d")
mkdir -p ./log/si/${today}
rtype='Lasso'
# rtype='Naive'
# rtype='Ridge'
# rtype='normal'

# rtype='MIOSR'
# rtype='SSR'
# rtype='STLSQ'

# evaluate=$1
# for data_name in 'lotka_volterra' 'hopf' 'cubic' 'vanderpol' 'rossler'
# do
# echo "------------------"
# echo "data:" $data_name
# echo "------------------"
logfn="./log/si/${today}/SI_eval_${rtype}.txt"
# logfn="./log/si/SI_eval_syn_${today}.txt"
# for rtype in 'MIOSR' 'SSR' 'STLSQ'
# nohup poetry run python -W ignore _SI_sindy.py --num_works $num_works --date $today --rtype $rtype> $logfn &
nohup poetry run python -W ignore _SI.py --num_works $num_works --date $today --rtype $rtype> $logfn &
# done