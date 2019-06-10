wget http://download.tensorflow.org/example_images/flower_photos.tgz
tar zxf flower_photos.tgz
ln -s flower_photos images
python prepare_data.py
python gen_tfrecord.py
