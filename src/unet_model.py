import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, AveragePooling2D, SpatialDropout2D, LeakyReLU, Dropout
from tensorflow.keras.models import Model

# U-net architecture inspired from following link: https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

def convolutional_block(input, num_filters, dropout=False):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    
    if dropout:
        x = Dropout(0.5)(x, training=True)
    return x

def encoder(input, num_filters,pool_size, dropout=False):
    x = convolutional_block(input, num_filters)
    p = MaxPool2D((pool_size, pool_size))(x)
    
    if dropout:
        p = Dropout(0.5)(p, training=True)
        
    return x, p

def decoder(input, skip_features, num_filters, strides,dropout=False):
    
    x = Conv2DTranspose(num_filters, (3, 3), strides=strides, padding="same")(input)
    
    x = Concatenate()([x, skip_features])
        
    x = convolutional_block(x, num_filters)
    
    if dropout:
        x = Dropout(0.5)(x, training=True)
        
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)
        
    s1, p1 = encoder(inputs, 8, pool_size=3)
    
    s2, p2 = encoder(p1, 16, pool_size=3, dropout=False)
    
    s3, p3 = encoder(p2, 32, pool_size=2, dropout=False)
    
    b1 = convolutional_block(p3, 64, dropout=False)
    
    d1 = decoder(b1, s3, 32,strides=2, dropout=False)
    
    d2 = decoder(d1, s2, 16,strides=3, dropout=False)
        
    outputs = Conv2D(1, 1, padding="same", activation="linear")(d2)
    
    model = Model(inputs, outputs, name="U-Net")
    return model
