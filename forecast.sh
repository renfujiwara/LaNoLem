#--- mocap: chicken dance
stream="./data/mocap/21_01.amc.4d"
outdir="./result/forecast/mocap_21_01/"
lstep=30
#--- mocap: exercise
#stream="./data/mocap/86_11.amc.4d"
#outdir="./result/forecast/mocap_86_01/"
#lstep=100
#--- google: social media
#stream="./data/google/socialmedia_g_ma4"
#outdir="./result/forecast/gt_socialmedia/"
#lstep=13 #3month-ahead

mkdir -p $outdir
echo "------------------"
echo "START: "
echo  $stream
echo  $outdir
echo "------------------"
#echo "Please see log.txt >>>  " $outdir"log.txt"

# without modelDB (please use this)
python -W ignore _Forecast.py -s $stream  -o $outdir -l $lstep #> $outdir"log.txt"

# with modelDB (if modelDB is given)
#modelDBfn=$outdir"MDBH.obj"
#python -W ignore main_rc.py -s $stream  -o $outdir -l $lstep -q $modelDBfn -i "no" #> $outdir"logi.txt"

echo "=================="
echo " RegimeCast END   "
echo "=================="




