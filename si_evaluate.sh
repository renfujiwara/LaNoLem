num_works=-1

today=$(date "+%Y%m%d")
mkdir -p ./log/si/${today}
# rtype='Lasso'
# rtype='Naive'
# rtype='Ridge'
rtype='normal'

logfn="./log/si/${today}/SI_eval_${rtype}.txt"
nohup poetry run python -W ignore _SI.py --num_works $num_works --date $today --rtype $rtype> $logfn &


# rtype='MIOSR'
# rtype='SSR'
# rtype='STLSQ'
# logfn="./log/si/SI_eval_syn_${today}.txt"
# nohup poetry run python -W ignore _SI_sindy.py --num_works $num_works --date $today --rtype $rtype> $logfn &
# done