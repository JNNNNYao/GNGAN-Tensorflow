import tensorflow as tf
from tensorflow.keras import layers as tfkl


class Generator(tf.Module):
    def __init__(self, z_dim, M=4):
        super().__init__()
        self.M = M
        self.linear = tfkl.Dense(M * M * 512, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))
        self.main = tf.keras.Sequential([
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(256, 4, strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(128, 4, strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(64, 4, strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2DTranspose(3, 3, strides=1, padding="same", activation='tanh',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])

    def __call__(self, z, *args, **kwargs):
        x = self.linear(z)
        x = tf.reshape(x, [-1, self.M, self.M, 512])
        x = self.main(x)
        return x


class Discriminator(tf.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = tf.keras.Sequential([
            tfkl.Conv2D(64, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.LeakyReLU(0.1),
            tfkl.Conv2D(128, 4, strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.LeakyReLU(0.1),
            tfkl.Conv2D(128, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.LeakyReLU(0.1),
            tfkl.Conv2D(256, 4, strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.LeakyReLU(0.1),
            tfkl.Conv2D(256, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.LeakyReLU(0.1),
            tfkl.Conv2D(512, 4, strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.LeakyReLU(0.1),
            tfkl.Conv2D(512, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.LeakyReLU(0.1),
            tfkl.Flatten()
        ])

        self.linear = tfkl.Dense(1, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))

    def __call__(self, x, *args, **kwargs):
        x = self.main(x)
        x = self.linear(x)
        return x


class Generator32(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=4)


class Generator48(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=6)


class Generator64(Generator):
    def __init__(self, z_dim, *args):
        super().__init__(z_dim, M=8)


class Discriminator32(Discriminator):
    def __init__(self, *args):
        super().__init__(M=32)


class Discriminator48(Discriminator):
    def __init__(self, *args):
        super().__init__(M=48)


class Discriminator64(Discriminator):
    def __init__(self, *args):
        super().__init__(M=64)


if __name__ == '__main__':
    G = Generator32(128)
    D = Discriminator32()
    z = tf.random.normal((16, 128))
    fake32 = G(z)
    pred32 = D(fake32)
    print(fake32.shape)
    print(pred32.shape)
    G = Generator48(128)
    D = Discriminator48()
    z = tf.random.normal((16, 128))
    fake48 = G(z)
    pred48 = D(fake48)
    print(fake48.shape)
    print(pred48.shape)
    G = Generator64(128)
    D = Discriminator64()
    z = tf.random.normal((16, 128))
    fake64 = G(z)
    pred64 = D(fake64)
    print(fake64.shape)
    print(pred64.shape)
