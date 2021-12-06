import os
import argparse
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from models import dcgan, resnet


net_G_models = {
    'dcgan.32': dcgan.Generator32,
    'dcgan.48': dcgan.Generator48,
    'dcgan.64': dcgan.Generator64,
    'resnet.32': resnet.ResGenerator32,
    'resnet.48': resnet.ResGenerator48,
    'resnet.128': resnet.ResGenerator128,
    'resnet.256': resnet.ResGenerator256,
}


def get_arguments():
    parser = argparse.ArgumentParser(description="DEMO")
    parser.add_argument("--arch", type=str, default="dcgan.64", choices=net_G_models.keys(), help="options: {}".format(net_G_models.keys()))
    parser.add_argument("--ckpt", type=str, default="./logdir/model", help="ckeckpoint path")
    parser.add_argument("--z_dim", type=int, default=128, help="latent space dimension")
    parser.add_argument("--num_images", type=int, default=10, help="# images for demo")
    parser.add_argument("--interpolate", action='store_true', help="generate image with interpolate latent")
    parser.add_argument("--condition", action='store_true', help="use condition GAN")
    parser.add_argument("--c", type=int, default=3, help="class to generate")
    parser.add_argument("--n_classes", type=int, default=3, help="# classes for condition GAN")
    parser.add_argument("--logdir", type=str, default="./logdir", help="path to save demo images")
    return parser.parse_args()


@tf.function
def G_infer(z):
    return net_G(z)


def interpolate_hypersphere(v1, v2, num_steps):
    v1_norm = tf.norm(v1)
    v2_norm = tf.norm(v2)
    v2_normalized = v2 * (v1_norm / v2_norm)

    vectors = []
    for step in range(num_steps):
        interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
        interpolated_norm = tf.norm(interpolated)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        vectors.append(interpolated_normalized)
    return tf.stack(vectors)


def interpolate_between_vectors():
    v1 = tf.random.normal([args.z_dim])
    v2 = tf.random.normal([args.z_dim])

    vectors = interpolate_hypersphere(v1, v2, args.num_images)

    return vectors


def demo():
    if args.interpolate:
        vectors = interpolate_between_vectors()
    else:
        vectors = tf.random.normal([args.num_images, args.z_dim])
    
    if args.condition:
        label = tf.one_hot(([args.c]*args.num_images), args.n_classes)
        vectors = tf.concat([vectors, label], -1)

    model_ckpt = tf.train.Checkpoint(net_G=net_G)
    model_manager = tf.train.CheckpointManager(model_ckpt, args.ckpt, max_to_keep=1)
    if model_manager.latest_checkpoint:
        model_ckpt.restore(model_manager.latest_checkpoint)
        print('{} restored!!'.format(model_manager.latest_checkpoint))
    else:
        print('checkpoint not found!!')
        exit()

    images = G_infer(vectors)
    images = ((images + 1) / 2) * 255
    for i in range(args.num_images):
        img = Image.fromarray(images[i].numpy().astype(np.uint8))
        img.save(os.path.join(args.logdir, "{}.jpg".format(i+1)))


if __name__ == '__main__':
    args = get_arguments()
    os.makedirs(args.logdir, exist_ok=True)

    # model
    net_G = net_G_models[args.arch](args.z_dim)

    demo()


# python3 demo.py --ckpt=logdir/car/model/ --logdir=output/Car
# python3 demo.py --ckpt=logdir/car/model/ --logdir=output/Interpolate --interpolate
# python3 demo.py --ckpt=logdir/cGAN/model/ --logdir=output/Audi --condition --c=0 --n_classes=3
# python3 demo.py --ckpt=logdir/cGAN/model/ --logdir=output/BMW --condition --c=1 --n_classes=3
# python3 demo.py --ckpt=logdir/cGAN/model/ --logdir=output/Mercedes-Benz --condition --c=2 --n_classes=3