import os
import json
import argparse
from tqdm import trange
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from models import dcgan, resnet
from datasets import get_dataset
from losses import BCEWithLogits


net_G_models = {
    'dcgan.32': dcgan.Generator32,
    'dcgan.48': dcgan.Generator48,
    'dcgan.64': dcgan.Generator64,
    'resnet.32': resnet.ResGenerator32,
    'resnet.48': resnet.ResGenerator48,
    'resnet.128': resnet.ResGenerator128,
    'resnet.256': resnet.ResGenerator256,
}


net_D_models = {
    'dcgan.32': dcgan.Discriminator32,
    'dcgan.48': dcgan.Discriminator48,
    'dcgan.64': dcgan.Discriminator64,
    'resnet.32': resnet.ResDiscriminator32,
    'resnet.48': resnet.ResDiscriminator48,
    'resnet.128': resnet.ResDiscriminator128,
    'resnet.256': resnet.ResDiscriminator256,
}


loss_fns = {
    'bce': BCEWithLogits,
}


datasets = ['car_brand']


def get_arguments():
    parser = argparse.ArgumentParser(description="GNGAN")
    parser.add_argument("--resume", action='store_true', help="resume from checkpoint")
    # model and training
    parser.add_argument("--dataset", type=str, default="car_brand", choices=datasets, help="options: {}".format(datasets))
    parser.add_argument("--rootpath", type=str, default="./confirmed_fronts", help="path to dataset folder")
    parser.add_argument("--arch", type=str, default="dcgan.64", choices=net_G_models.keys(), help="options: {}".format(net_G_models.keys()))
    parser.add_argument("--loss", type=str, default="bce", choices=loss_fns.keys(), help="options: {}".format(loss_fns.keys()))
    parser.add_argument("--total_steps", type=int, default=200000, help="total number of training steps")
    parser.add_argument("--batch_size_D", type=int, default=64, help="batch size for discriminator")
    parser.add_argument("--batch_size_G", type=int, default=128, help="batch size for generator")
    parser.add_argument("--lr_D", type=float, default=2e-4, help="Discriminator learning rate")
    parser.add_argument("--lr_G", type=float, default=2e-4, help="Generator learning rate")
    parser.add_argument("--beta_1", type=float, default=0.0, help="for Adam")
    parser.add_argument("--beta_2", type=float, default=0.9, help="for Adam")
    parser.add_argument("--n_dis", type=int, default=1, help="update Generator every this steps")
    parser.add_argument("--z_dim", type=int, default=128, help="latent space dimension")
    # logging
    parser.add_argument("--sample_step", type=int, default=1000, help="sample image every this steps")
    parser.add_argument("--sample_size", type=int, default=24, help="sampling size of images")
    parser.add_argument("--save_step", type=int, default=5000, help="save model every this step")
    parser.add_argument("--num_images", type=int, default=16384, help="# images for evaluation")
    parser.add_argument("--logdir", type=str, default="./logdir", help="log folder")    
    return parser.parse_args()
    

def make_grid(sample, idx):
    plt.figure(figsize=(10,4))
    for i in range(24):
        plt.subplot(3, 8, i+1)
        plt.imshow(sample[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(args.logdir, 'sample', 'image_{:d}.png'.format(idx)))
    plt.show()


@tf.function
def G_infer(z, label):
    return net_G(tf.concat([z, label], -1))


@tf.function
def train_D_step(images, label, wrong_label):
    z = tf.random.normal((args.batch_size_D, args.z_dim))

    with tf.GradientTape() as disc_tape, tf.GradientTape() as gn_tape:
        fake_images = tf.stop_gradient(net_G(tf.concat([z, label], -1)))
        # discriminator will get 3 kinds of pair:
        # (real_img, real_label), (fake_img, real_label), (real_img, wrong_label)
        # only the first pair is valid pair
        label_map = label[:, tf.newaxis, tf.newaxis, :] * tf.ones((images.shape[0], images.shape[1], images.shape[2], 3))
        wrong_label_map = wrong_label[:, tf.newaxis, tf.newaxis, :] * tf.ones((images.shape[0], images.shape[1], images.shape[2], 3))
        pair1 = tf.concat([images, label_map], -1)
        pair2 = tf.concat([fake_images, label_map], -1)
        pair3 = tf.concat([images, wrong_label_map], -1)
        D_input = tf.concat([pair1, pair2, pair3], 0)
        gn_tape.watch(D_input)
        y = net_D(D_input)

        # gradnorm
        grad = gn_tape.gradient(y, D_input)
        grad_norm = tf.norm(tf.reshape(grad, (tf.shape(grad)[0], -1)), ord=2, axis=1)
        grad_norm = grad_norm[:, tf.newaxis]
        pred = (y / (grad_norm + tf.abs(y)))

        pred_real, pred_fake = pred[:args.batch_size_D], pred[args.batch_size_D:]
        loss, loss_real, loss_fake = loss_fn(pred_real, pred_fake)

    gradients_of_D = disc_tape.gradient(loss, net_D.trainable_variables)
    optim_D.apply_gradients(zip(gradients_of_D, net_D.trainable_variables))
    return loss, loss_real, loss_fake


@tf.function
def train_G_step(label):
    z = tf.random.normal((args.batch_size_G, args.z_dim))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as gn_tape:
        fake_images = net_G(tf.concat([z, label], -1))
        label_map = label[:, tf.newaxis, tf.newaxis, :] * tf.ones((fake_images.shape[0], fake_images.shape[1], fake_images.shape[2], 3))
        D_input = tf.concat([fake_images, label_map], -1)
        gn_tape.watch(D_input)
        y = net_D(D_input)

        # gradnorm
        grad = gn_tape.gradient(y, D_input)
        grad_norm = tf.norm(tf.reshape(grad, (tf.shape(grad)[0], -1)), ord=2, axis=1)
        grad_norm = grad_norm[:, tf.newaxis]
        pred_fake = (y / (grad_norm + tf.abs(y)))

        loss = loss_fn(pred_fake)

    gradients_of_G = gen_tape.gradient(loss, net_G.trainable_variables)
    optim_G.apply_gradients(zip(gradients_of_G, net_G.trainable_variables))
    return loss


def train():
    # fixed z
    fixed_z = tf.random.normal((args.sample_size, args.z_dim))
    fixed_z = tf.Variable(fixed_z)  # trackable for tf.train.Checkpoint
    # fixed z
    fixed_label = tf.one_hot(([0]*8 + [1]*8 + [2]*8), 3)
    fixed_label = tf.Variable(fixed_label)  # trackable for tf.train.Checkpoint

    writer = tf.summary.create_file_writer(str(args.logdir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()

    model_path = os.path.join(args.logdir, 'model')
    model_ckpt = tf.train.Checkpoint(net_G=net_G)
    model_manager = tf.train.CheckpointManager(model_ckpt, model_path, max_to_keep=10)
    
    checkpoint_path = os.path.join(args.logdir, 'checkpoints')
    ckpt = tf.train.Checkpoint(net_G=net_G, net_D=net_D, optim_G=optim_G, optim_D=optim_D, fixed_z=fixed_z, fixed_label=fixed_label)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
    if args.resume:
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('{} restored!!'.format(ckpt_manager.latest_checkpoint))
        else:
            print('checkpoint not found!!')
    else:
        os.makedirs(os.path.join(args.logdir, 'sample'), exist_ok=True)
        os.makedirs(os.path.join(args.logdir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(args.logdir, 'checkpoints'), exist_ok=True)

    with trange(1, args.total_steps + 1, ncols=0, initial=0, total=args.total_steps) as pbar:
        for step in pbar:
            loss_sum = 0
            loss_real_sum = 0
            loss_fake_sum = 0

            x, labels = next(dataset)
            wrong_labels = tf.math.floormod(labels + tf.random.uniform(labels.shape, 1, 3, dtype=tf.int32), 3)
            labels = tf.one_hot(tf.reshape(labels, (x.shape[0])), 3)
            wrong_labels = tf.one_hot(tf.reshape(wrong_labels, (x.shape[0])), 3)
            x = iter(tf.split(x, num_or_size_splits=args.n_dis))
            labels = iter(tf.split(labels, num_or_size_splits=args.n_dis))
            wrong_labels = iter(tf.split(wrong_labels, num_or_size_splits=args.n_dis))
            # Discriminator
            for _ in range(args.n_dis):
                x_real = next(x)
                label = next(labels)
                wrong_label = next(wrong_labels)
                loss, loss_real, loss_fake = train_D_step(x_real, label, wrong_label)

                loss_sum += loss
                loss_real_sum += loss_real
                loss_fake_sum += loss_fake

            loss = loss_sum / args.n_dis
            loss_real = loss_real_sum / args.n_dis
            loss_fake = loss_fake_sum / args.n_dis

            pbar.set_postfix(loss_real='%.3f' % loss_real, loss_fake='%.3f' % loss_fake)

            # Generator
            label = tf.random.uniform((args.batch_size_G,), 0, 3, dtype=tf.int32)
            label = tf.one_hot(label, 3)
            loss_G = train_G_step(label)

            # write summaries
            tf.summary.scalar('Discriminator/loss', loss, step=step)
            tf.summary.scalar('Discriminator/loss_real', loss_real, step=step)
            tf.summary.scalar('Discriminator/loss_fake', loss_fake, step=step)
            tf.summary.scalar('Generator/loss', loss_G, step=step)
            writer.flush()

            # sample from fixed z
            if step == 1 or step % args.sample_step == 0:
                sample = G_infer(fixed_z, fixed_label)
                sample = (sample + 1) / 2
                make_grid(sample, step)

            if step == 1 or step % args.save_step == 0:
                ckpt_save_path = ckpt_manager.save()
                model_manager.save()
                pbar.write(f'Step: {step}, save checkpoint at {ckpt_save_path}')

            k = len(str(args.total_steps))
            pbar.write(f"{step:{k}d}/{args.total_steps} ")

    writer.close()


if __name__ == '__main__':
    args = get_arguments()
    os.makedirs(args.logdir, exist_ok=True)
    with open('{}/config.json'.format(args.logdir), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # model
    net_G = net_G_models[args.arch](args.z_dim)
    net_D = net_D_models[args.arch]()

    # loss
    loss_fn = loss_fns[args.loss]

    # optimizer
    optim_G = tf.keras.optimizers.Adam(learning_rate=args.lr_G, beta_1=args.beta_1, beta_2=args.beta_2)
    optim_D = tf.keras.optimizers.Adam(learning_rate=args.lr_D, beta_1=args.beta_1, beta_2=args.beta_2)

    # dataset
    dataset = get_dataset(args.dataset, args.rootpath, args.batch_size_D * args.n_dis)

    train()
