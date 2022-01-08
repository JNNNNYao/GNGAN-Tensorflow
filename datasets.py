import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_gan as tfgan


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def preprocessing_car(image_path, label=0):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))
    img = tf.image.random_flip_left_right(img)
    img = img / 127.5 - 1
    return img, [label]


def car(path, bs, training=True):
    paths = recursive_glob(rootdir=path, suffix=".jpg")
    random.Random(24).shuffle(paths)
    print("Found {} images".format(len(paths)))

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(bs, drop_remainder=training)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.repeat()
    return dataset


def car_brand(path, bs):
    # 3 classes: Audi, BMW, Mercedes-Benz
    paths_Audi = recursive_glob(rootdir=os.path.join(path, 'Audi'), suffix=".jpg")
    num_Audi = len(paths_Audi)
    labels_Audi = [0]*num_Audi
    paths_BMW = recursive_glob(rootdir=os.path.join(path, 'BMW'), suffix=".jpg")
    num_BMW = len(paths_BMW)
    labels_BMW = [1]*num_BMW
    paths_Benz = recursive_glob(rootdir=os.path.join(path, 'Mercedes-Benz'), suffix=".jpg")
    num_Benz = len(paths_Benz)
    labels_Benz = [2]*num_Benz
    print("Found {} images: {}/{}/{}".format((num_Audi + num_BMW + num_Benz), num_Audi, num_BMW, num_Benz))

    ds_Audi = tf.data.Dataset.from_tensor_slices((paths_Audi, labels_Audi))
    ds_Audi = ds_Audi.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_Audi = ds_Audi.shuffle(10000)
    ds_Audi = ds_Audi.repeat()

    ds_BMW = tf.data.Dataset.from_tensor_slices((paths_BMW, labels_BMW))
    ds_BMW = ds_BMW.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_BMW = ds_BMW.shuffle(10000)
    ds_BMW = ds_BMW.repeat()

    ds_Benz = tf.data.Dataset.from_tensor_slices((paths_Benz, labels_Benz))
    ds_Benz = ds_Benz.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_Benz = ds_Benz.shuffle(10000)
    ds_Benz = ds_Benz.repeat()

    dataset = tf.data.Dataset.sample_from_datasets([ds_Audi, ds_BMW, ds_Benz], weights=[1/3, 1/3, 1/3])
    dataset = dataset.batch(bs, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def car_brand_color(path, bs):
    # 4 classes: Audi_white, Audi_black, BMW_white, BMW_black
    paths_Audi_white = [path for path in recursive_glob(rootdir=os.path.join(path, 'Audi'), suffix=".jpg") if 'White' in path]
    num_Audi_white = len(paths_Audi_white)
    labels_Audi_white = [0]*num_Audi_white
    paths_Audi_black = [path for path in recursive_glob(rootdir=os.path.join(path, 'Audi'), suffix=".jpg") if 'Black' in path]
    num_Audi_black = len(paths_Audi_black)
    labels_Audi_black = [1]*num_Audi_black
    paths_BMW_white = [path for path in recursive_glob(rootdir=os.path.join(path, 'BMW'), suffix=".jpg") if 'White' in path]
    num_BMW_white = len(paths_BMW_white)
    labels_BMW_white = [2]*num_BMW_white
    paths_BMW_black = [path for path in recursive_glob(rootdir=os.path.join(path, 'BMW'), suffix=".jpg") if 'Black' in path]
    num_BMW_black = len(paths_BMW_black)
    labels_BMW_black = [3]*num_BMW_black
    print("Found {} images: {}/{}/{}/{}".format((num_Audi_white + num_Audi_black + num_BMW_white + num_BMW_black), num_Audi_white, num_Audi_black, num_BMW_white, num_BMW_black))

    ds_Audi_white = tf.data.Dataset.from_tensor_slices((paths_Audi_white, labels_Audi_white))
    ds_Audi_white = ds_Audi_white.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_Audi_white = ds_Audi_white.shuffle(2000)
    ds_Audi_white = ds_Audi_white.repeat()

    ds_Audi_black = tf.data.Dataset.from_tensor_slices((paths_Audi_black, labels_Audi_black))
    ds_Audi_black = ds_Audi_black.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_Audi_black = ds_Audi_black.shuffle(2000)
    ds_Audi_black = ds_Audi_black.repeat()

    ds_BMW_white = tf.data.Dataset.from_tensor_slices((paths_BMW_white, labels_BMW_white))
    ds_BMW_white = ds_BMW_white.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_BMW_white = ds_BMW_white.shuffle(2000)
    ds_BMW_white = ds_BMW_white.repeat()

    ds_BMW_black = tf.data.Dataset.from_tensor_slices((paths_BMW_black, labels_BMW_black))
    ds_BMW_black = ds_BMW_black.map(preprocessing_car, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_BMW_black = ds_BMW_black.shuffle(2000)
    ds_BMW_black = ds_BMW_black.repeat()

    dataset = tf.data.Dataset.sample_from_datasets([ds_Audi_white, ds_Audi_black, ds_BMW_white, ds_BMW_black], weights=[1/4, 1/4, 1/4, 1/4])
    dataset = dataset.batch(bs, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def preprocessing_cifar10(img, label):
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)
    img = img / 127.5 - 1
    return img, label


def cifar10(bs, training=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.map(preprocessing_cifar10, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(bs, drop_remainder=training)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.repeat()
    return dataset


INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_FINAL_POOL = 'pool_3'
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def calc_statistic(data):
    size = INCEPTION_DEFAULT_IMAGE_SIZE
    classifier_fn = tfgan.eval.classifier_fn_from_tfhub(INCEPTION_TFHUB, INCEPTION_FINAL_POOL, True)
    real_data_act = []
    for img, _ in iter(data):
        img = tf.image.resize(img, [size, size], method=tf.image.ResizeMethod.BILINEAR)
        real_data_act.append(classifier_fn(img))
    real_data_act = tf.concat(real_data_act, 0)
    return real_data_act


def get_dataset(ds, path, bs):
    if ds == 'car':
        dataset, test_dataset = car(path, bs), car(path, 1024, False)
        real_data_act = calc_statistic(test_dataset)
        return iter(dataset), real_data_act

    if ds == 'car_brand':
        dataset = car_brand(path, bs)
        return iter(dataset)

    if ds == 'car_brand_color':
        dataset = car_brand_color(path, bs)
        return iter(dataset)

    if ds == 'cifar10':
        dataset, test_dataset = cifar10(bs), cifar10(1024, False)
        real_data_act = calc_statistic(test_dataset)
        return iter(dataset), real_data_act


if __name__ == '__main__':
    dataset, real_data_act = get_dataset('car', './confirmed_fronts', 32)
    images, labels = next(dataset)
    print(images.shape)
    print(labels.shape)
    print(real_data_act.shape)
    dataset = get_dataset('car_brand', './confirmed_fronts', 32)
    images, labels = next(dataset)
    print(images.shape)
    print(labels.shape)
    dataset = get_dataset('car_brand_color', './confirmed_fronts', 32)
    images, labels = next(dataset)
    print(images.shape)
    print(labels.shape)
    dataset, real_data_act = get_dataset('cifar10', './', 32)
    images, labels = next(dataset)
    print(images.shape)
    print(labels.shape)
    print(real_data_act.shape)