from tensorflow.keras.applications import EfficientNetB7
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization,Activation, Input,GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

def get_EffcientNet_model(input_shape,num_classes):
    base_model = EfficientNetB7(weights='imagenet', 
                        include_top=False, 
                        input_shape=input_shape, 
                        classes=num_classes)
    
    inputs = Input(shape=input_shape)
    
    x = base_model(inputs)
    x = GlobalAveragePooling2D()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(512, activation='relu')(x) # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dropout(0.25)(x) # to avoid overfitting
    x = Dense(256, activation='relu')(x) # dense layer 2
    #x = tf.keras.layers.BatchNormalization()(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.layers[1].trainable = False
    
    return model

    
    
if __name__ == '__main__':
    DATA_SIZE=10000
    BASE_PATH=f'data/{DATA_SIZE}'
    data=load_train_data(DATA_SIZE,'final_df.csv')
