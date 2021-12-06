import tensorflow as tf


def BCEWithLogits(pred_real, pred_fake=None):
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    if pred_fake is not None:
        loss_real = loss_fn(tf.ones_like(pred_real), pred_real)
        loss_fake = loss_fn(tf.zeros_like(pred_fake), pred_fake)
        loss = loss_real + loss_fake
        return loss, loss_real, loss_fake
    else:
        loss = loss_fn(tf.ones_like(pred_real), pred_real)
        return loss


if __name__ == '__main__':
    from models import dcgan
    G = dcgan.Generator32(128)
    D = dcgan.Discriminator32()
    z = tf.random.normal((2, 128))
    fake = G(z)
    pred = D(fake)
    print(BCEWithLogits(pred, pred))
    print(BCEWithLogits(pred))