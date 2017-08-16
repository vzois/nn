DEVICE=gpu

python tf_cnn_benchmarks.py train $DEVICE --model=alexnet --batch_size=16 > alexnet_train
python tf_cnn_benchmarks.py train $DEVICE --model=googlenet --batch_size=16 > googlenet_train
python tf_cnn_benchmarks.py train $DEVICE --model=inception3 --batch_size=16 > inception3_train
python tf_cnn_benchmarks.py train $DEVICE --model=inception4 --batch_size=16 > inception4_train
python tf_cnn_benchmarks.py train $DEVICE --model=resnet50 --batch_size=16 > resnet50_train
python tf_cnn_benchmarks.py train $DEVICE --model=resnet101 --batch_size=16 > resnet101_train
python tf_cnn_benchmarks.py train $DEVICE --model=resnet152 --batch_size=16 > resnet152_train
python tf_cnn_benchmarks.py train $DEVICE --model=vgg11 --batch_size=16 > vgg11_train
python tf_cnn_benchmarks.py train $DEVICE --model=vgg16 --batch_size=16 > vgg16_train
python tf_cnn_benchmarks.py train $DEVICE --model=vgg19 --batch_size=16 > vgg19_train

python tf_cnn_benchmarks.py classify $DEVICE --model=alexnet --batch_size=16 > alexnet_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=googlenet --batch_size=16 > googlenet_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=inception3 --batch_size=16 > inception3_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=inception4 --batch_size=16 > inception4_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=resnet50 --batch_size=16 > resnet50_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=resnet101 --batch_size=16 > resnet101_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=resnet152 --batch_size=16 > resnet152_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=vgg11 --batch_size=16 > vgg11_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=vgg16 --batch_size=16 > vgg16_classify
python tf_cnn_benchmarks.py classify $DEVICE --model=vgg19 --batch_size=16 > vgg19_classify