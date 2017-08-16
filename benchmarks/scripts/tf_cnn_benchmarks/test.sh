DEVICE=gpu
BATCH_SIZE=16
FORWARD=True

if [$DEVICE == "cpu"]
then
DATA_FORMAT=NHWC
else
DATA_FORMAT=NCHW
fi

models=("alexnet" "googlenet" "inception3" "inception4" "resnet50" "resnet101" "resnet152" "vgg11" "vgg16" "vgg19")

rm -rf *_train
rm -rf *_classify

for model in "${models[@]}"
do
	echo $model
	if [ $FORWARD == "True" ]
	then
		python tf_cnn_benchmarks.py --model=$model --batch_size=$BATCH_SIZE --device=$DEVICE --data_format=$DATA_FORMAT --forward_only=$FORWARD > $model'_classify'
	else
		python tf_cnn_benchmarks.py --model=$model --batch_size=$BATCH_SIZE --device=$DEVICE --data_format=$DATA_FORMAT --forward_only=$FORWARD > $model'_train'
	fi
done