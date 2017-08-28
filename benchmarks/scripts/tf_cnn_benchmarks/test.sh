DEVICE=gpu
BATCH_SIZE=512
FORWARD=True

if [ $DEVICE == "cpu" ]
then
DATA_FORMAT=NHWC
else
DATA_FORMAT=NHWC
fi

#MODELS TO CHOOSE FROM
ALEXNET=0
GOOGLENET=1
INCEPTION3=2
INCEPTION4=3
RESNET50=4
RESNET101=5
RESNET152=6
VGG11=7
VGG16=8
VGG19=9

#SET BENCH MODELS FOR BENCHMARKING
bench_model=($ALEXNET)
models=("alexnet" "googlenet" "inception3" "inception4" "resnet50" "resnet101" "resnet152" "vgg11" "vgg16" "vgg19")
batch_classify=(512 512 512 512 512 512 512 256 256 128)
batch_train=(512 256 64 64 64 64 64 128 128 128)

if [ $FORWARD == "True" ]
then
	rm -rf classify/*_classify
else
	rm -rf train/*_train
fi

for i in "${bench_model[@]}"
do
	model=${models[$i]}
	if [ $FORWARD == "True" ]
	then
		
		batch=${batch_classify[$i]}
		echo "classify>"$model":"$batch,$DEVICE
		python tf_cnn_benchmarks.py --model=$model --batch_size=$batch --device=$DEVICE --data_format=$DATA_FORMAT --forward_only=$FORWARD > 'classify/'$model'_classify'
	else
		
		batch=${batch_classify[$i]}
		echo "train>"$model":"$batch,$DEVICE
		python tf_cnn_benchmarks.py --model=$model --batch_size=$batch --device=$DEVICE --data_format=$DATA_FORMAT --forward_only=$FORWARD > 'train/'$model'_train'
	fi
done