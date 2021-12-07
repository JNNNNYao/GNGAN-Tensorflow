import tensorflow as tf
from tensorflow.keras import layers as tfkl


class GenBlock(tfkl.Layer):
    def __init__(self, in_channels, out_channels):
        super(GenBlock, self).__init__()
        # shortcut
        self.shortcut = tf.keras.Sequential([
            tfkl.UpSampling2D(size=(2, 2)),
            tfkl.Conv2D(out_channels, 1, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])
        # residual
        self.residual = tf.keras.Sequential([
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.UpSampling2D(size=(2, 2)),
            tfkl.Conv2D(out_channels, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(out_channels, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class ResGenerator32(tf.Module):
    def __init__(self, z_dim, *args):
        super().__init__()
        self.linear = tfkl.Dense(4 * 4 * 256, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))
        self.blocks = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=(4, 4, 256)),
            GenBlock(256, 256),
            GenBlock(256, 256),
            GenBlock(256, 256),
        ])
        self.output = tf.keras.Sequential([
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(3, 3, strides=1, padding="same", activation='tanh',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])

    def __call__(self, z, *args, **kwargs):
        z = self.linear(z)
        z = tf.reshape(z, [-1, 4, 4, 256])
        return self.output(self.blocks(z))


class ResGenerator48(tf.Module):
    def __init__(self, z_dim, *args):
        super().__init__()
        self.linear = tfkl.Dense(6 * 6 * 512, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))
        self.blocks = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=(6, 6, 512)),
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        ])
        self.output = tf.keras.Sequential([
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(3, 3, strides=1, padding="same", activation='tanh',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])

    def __call__(self, z, *args, **kwargs):
        z = self.linear(z)
        z = tf.reshape(z, [-1, 6, 6, 512])
        return self.output(self.blocks(z))


class ResGenerator64(tf.Module):
    def __init__(self, z_dim, *args):
        super().__init__()
        self.linear = tfkl.Dense(4 * 4 * 512, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))
        self.blocks = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=(4, 4, 512)),
            GenBlock(512, 512),
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        ])
        self.output = tf.keras.Sequential([
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(3, 3, strides=1, padding="same", activation='tanh',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])

    def __call__(self, z, *args, **kwargs):
        z = self.linear(z)
        z = tf.reshape(z, [-1, 4, 4, 512])
        return self.output(self.blocks(z))


class ResGenerator128(tf.Module):
    def __init__(self, z_dim, *args):
        super().__init__()
        self.linear = tfkl.Dense(4 * 4 * 1024, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))
        self.blocks = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=(4, 4, 1024)),
            GenBlock(1024, 1024),
            GenBlock(1024, 512),
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        ])
        self.output = tf.keras.Sequential([
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(3, 3, strides=1, padding="same", activation='tanh',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])

    def __call__(self, z, *args, **kwargs):
        z = self.linear(z)
        z = tf.reshape(z, [-1, 4, 4, 1024])
        return self.output(self.blocks(z))


class ResGenerator256(tf.Module):
    def __init__(self, z_dim, *args):
        super().__init__()
        self.linear = tfkl.Dense(4 * 4 * 1024, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))
        self.blocks = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=(4, 4, 1024)),
            GenBlock(1024, 1024),
            GenBlock(1024, 512),
            GenBlock(512, 512),
            GenBlock(512, 256),
            GenBlock(256, 128),
            GenBlock(128, 64),
        ])
        self.output = tf.keras.Sequential([
            tfkl.BatchNormalization(),
            tfkl.ReLU(),
            tfkl.Conv2D(3, 3, strides=1, padding="same", activation='tanh',
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])

    def __call__(self, z, *args, **kwargs):
        z = self.linear(z)
        z = tf.reshape(z, [-1, 4, 4, 1024])
        return self.output(self.blocks(z))


class OptimizedDisblock(tfkl.Layer):
    def __init__(self, in_channels, out_channels):
        super(OptimizedDisblock, self).__init__()
        # shortcut
        self.shortcut = tf.keras.Sequential([
            tfkl.AveragePooling2D(pool_size=(2, 2)),
            tfkl.Conv2D(out_channels, 1, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ])
        # residual
        self.residual = tf.keras.Sequential([
            tfkl.Conv2D(out_channels, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.ReLU(),
            tfkl.Conv2D(out_channels, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.AveragePooling2D(pool_size=(2, 2)),
        ])

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(tfkl.Layer):
    def __init__(self, in_channels, out_channels, down=False):
        super(DisBlock, self).__init__()
        # shortcut
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                tfkl.Conv2D(out_channels, 1, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)))
        if down:
            shortcut.append(tfkl.AveragePooling2D(pool_size=(2, 2)))
        self.shortcut = tf.keras.Sequential(shortcut)
        # residual
        residual = [
            tfkl.ReLU(),
            tfkl.Conv2D(out_channels, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
            tfkl.ReLU(),
            tfkl.Conv2D(out_channels, 3, strides=1, padding="same",
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0)),
        ]
        if down:
            residual.append(tfkl.AveragePooling2D(pool_size=(2, 2)))
        self.residual = tf.keras.Sequential(residual)

    def __call__(self, x):
        return (self.residual(x) + self.shortcut(x))


class ResDiscriminator32(tf.Module):
    def __init__(self, *args):
        super().__init__()
        self.model = tf.keras.Sequential([
            OptimizedDisblock(3, 128),
            DisBlock(128, 128, down=True),
            DisBlock(128, 128),
            DisBlock(128, 128),
            tfkl.ReLU(),
            tfkl.AveragePooling2D(pool_size=(1, 1)),
            tfkl.Flatten(),
        ])
        self.linear = tfkl.Dense(1, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))

    def __call__(self, x, *args, **kwargs):
        x = self.model(x)
        x = self.linear(x)
        return x


class ResDiscriminator48(tf.Module):
    def __init__(self, *args):
        super().__init__()
        self.model = tf.keras.Sequential([
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            tfkl.ReLU(),
            tfkl.AveragePooling2D(pool_size=(1, 1)),
            tfkl.Flatten(),
        ])
        self.linear = tfkl.Dense(1, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))

    def __call__(self, x, *args, **kwargs):
        x = self.model(x)
        x = self.linear(x)
        return x


class ResDiscriminator64(tf.Module):
    def __init__(self, *args):
        super().__init__()
        self.model = tf.keras.Sequential([
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            DisBlock(512, 512, down=True),
            tfkl.ReLU(),
            tfkl.AveragePooling2D(pool_size=(1, 1)),
            tfkl.Flatten(),
        ])
        self.linear = tfkl.Dense(1, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))

    def __call__(self, x, *args, **kwargs):
        x = self.model(x)
        x = self.linear(x)
        return x


class ResDiscriminator128(tf.Module):
    def __init__(self, *args):
        super().__init__()
        self.model = tf.keras.Sequential([
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            DisBlock(512, 1024, down=True),
            DisBlock(1024, 1024),
            tfkl.ReLU(),
            tfkl.AveragePooling2D(pool_size=(1, 1)),
            tfkl.Flatten(),
        ])
        self.linear = tfkl.Dense(1, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))

    def __call__(self, x, *args, **kwargs):
        x = self.model(x)
        x = self.linear(x)
        return x


class ResDiscriminator256(tf.Module):
    def __init__(self, *args):
        super().__init__()
        self.model = tf.keras.Sequential([
            OptimizedDisblock(3, 64),
            DisBlock(64, 128, down=True),
            DisBlock(128, 256, down=True),
            DisBlock(256, 512, down=True),
            DisBlock(512, 512, down=True),
            DisBlock(512, 1024, down=True),
            DisBlock(1024, 1024),
            tfkl.ReLU(),
            tfkl.AveragePooling2D(pool_size=(1, 1)),
            tfkl.Flatten(),
        ])
        self.linear = tfkl.Dense(1, 
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02), bias_initializer=tf.keras.initializers.Constant(0))

    def __call__(self, x, *args, **kwargs):
        x = self.model(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    G = ResGenerator32(128)
    D = ResDiscriminator32()
    z = tf.random.normal((2, 128))
    fake32 = G(z)
    pred32 = D(fake32)
    print(fake32.shape)
    print(pred32.shape)
    G = ResGenerator48(128)
    D = ResDiscriminator48()
    z = tf.random.normal((2, 128))
    fake48 = G(z)
    pred48 = D(fake48)
    print(fake48.shape)
    print(pred48.shape)
    G = ResGenerator64(128)
    D = ResDiscriminator64()
    z = tf.random.normal((2, 128))
    fake64 = G(z)
    pred64 = D(fake64)
    print(fake64.shape)
    print(pred64.shape)
    G = ResGenerator128(128)
    D = ResDiscriminator128()
    z = tf.random.normal((2, 128))
    fake128 = G(z)
    pred128 = D(fake128)
    print(fake128.shape)
    print(pred128.shape)
    G = ResGenerator256(128)
    D = ResDiscriminator256()
    z = tf.random.normal((2, 128))
    fake256 = G(z)
    pred256 = D(fake256)
    print(fake256.shape)
    print(pred256.shape)