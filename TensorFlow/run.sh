if [[ $# < 2 ]]; then
  echo Usage: $0 [number of threads] [benchmark-prefix]
  exit
fi


export KMP_AFFINITY='granularity=fine,compact,1,0'
export NUM_INTER_THREADS=1 
export NUM_INTRA_THREADS=$1
export OMP_NUM_THREADS=$1
#taskset -c 0-$(( $1 - 1 ))  python inference_benchmark.py 1 1> benchmark-results/${2}-${1}.txt
taskset -c 0-$(( $1 - 1 )),48-$(( $1 + 47 ))  python inference_benchmark.py 1 1> benchmark-results/${2}-${1}.txt


#echo KMP_BLOCKTIME     $KMP_BLOCKTIME
#echo NUM_INTER_THREADS $NUM_INTER_THREADS
#echo NUM_INTRA_THREADS $NUM_INTRA_THREADS
#echo OMP_NUM_THREADS   $OMP_NUM_THREADS
#taskset -c 24-$(( $1 + 23 )) python inference_benchmark.py 1 1> benchmark-results/${2}-${1}.txt
