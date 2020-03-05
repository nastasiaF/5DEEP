
# Neural Networks : Exploration du Mnist

## Import the libraries


```python
from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense, Dropout, Activation
from keras.models import Sequential
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import Model

from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

    Using TensorFlow backend.
    

## Load the data

Charger les données, elles sont déja dans la library keras

##### Question : Pourquoi utiliser 2 jeux de données : test et train ?



```python
(X_train, y_train), (X_test, y_test) = mnist.load_data() ### fashion_mnist.load_data
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (60000, 28, 28)
    (60000,)
    (10000, 28, 28)
    (10000,)
    

__Question__ : 
* Commbien d'image dans les jeux de données ? 
* Quelles est le type de probleme à resoudre ? 




```python
#Print the image :
plt.imshow(X_train[0])
```




    <matplotlib.image.AxesImage at 0x2138c8566d8>




![png](output_8_1.png)



```python
print( y_train[0])
```

    5
    


```python
y_train
```




    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)



## Pre-processing

__Question__ : 
* Pourquoi declarer une dimension supplémentaire ? 
* Pourquoi diviser par 255? 
* Comment la réponse est transformée ? Pourquoi?


```python
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)
```


```python
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
```


```python
num_classes = 10
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test, num_classes)
```


```python
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```

    (60000, 28, 28, 1)
    (60000, 10)
    (10000, 28, 28, 1)
    (10000, 10)
    


```python
print(y_train)
```

    [[0. 0. 0. ... 0. 0. 0.]
     [1. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 1. 0.]]
    

## Définir le model 

__Model Sequential__ : Modèle d'ajout de couche linéaire. 


```python
model = Sequential()
```


```python
model.add(Conv2D(32, kernel_size=(5,5),input_shape=(28,28,1), padding='same', activation='relu'))
```

Le premier layer est un produit de convolution 2D, que fait se layer?


```python
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(5,5),padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(10,activation='softmax'))

```


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 28, 28, 32)        832       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 14, 14, 64)        51264     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 7, 7, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3136)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              3212288   
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                10250     
    =================================================================
    Total params: 3,274,634
    Trainable params: 3,274,634
    Non-trainable params: 0
    _________________________________________________________________
    

### Comment voir se qui se passe ? 

Nous allons regarder couches par couche les sorties. 

https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer


```python
## Sorties des layers couches par couches
#Conv2D
intermediate_layer_1 = Model(inputs=model.input,
                                 outputs=model.get_layer('conv2d_1').output)
# Max Pooling 1 
intermediate_layer_2 = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_1').output)

#Conv2D
intermediate_layer_3 = Model(inputs=model.input, outputs=model.get_layer('conv2d_2').output)

# Max Pooling 2 
intermediate_layer_4 = Model(inputs=model.input, outputs=model.get_layer('max_pooling2d_2').output)

# Flatten
intermediate_layer_5 = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)

# Dense 
intermediate_layer_6 = Model(inputs=model.input, outputs=model.get_layer('dense_1').output)

# Dense 2
intermediate_layer_7 = Model(inputs=model.input, outputs=model.get_layer('dense_2').output)

```


```python
#Entrée 
data = X_train[0:1]
#data[0,:,:,0].shape
#data[0,:,:,0]
#np.savetxt('ENTRY_0.txt',np.asarray(data[0,:,:,0]))
plt.imshow(np.asarray(data[0,:,:,0]))
```




    <matplotlib.image.AxesImage at 0x2138ea19eb8>




![png](output_27_1.png)



```python
### Prediction sortie des couches intermédiares :
intermediate_output_1 = intermediate_layer_1.predict(data)
intermediate_output_2 = intermediate_layer_2.predict(data)
intermediate_output_3 = intermediate_layer_3.predict(data)
intermediate_output_4 = intermediate_layer_4.predict(data)
intermediate_output_5 = intermediate_layer_5.predict(data)
intermediate_output_6 = intermediate_layer_6.predict(data)
intermediate_output_7 = intermediate_layer_7.predict(data)
```

### Couche 1 : Conv2D

__Question__ : 
* Que fait cette couche? 
* Quel est l'intérêt de cette étape? 


```python
intermediate_output_1.shape
```




    (1, 28, 28, 32)




```python
output1 = intermediate_output_1[0,:,:,0]
output2 = intermediate_output_1[0,:,:,1]
output32 = intermediate_output_1[0,:,:,31]
#np.savetxt('LAYERS1_CONV_0.txt',np.asarray(output1))
#np.savetxt('LAYERS1_CONV_0.txt',np.asarray(output2))
```


```python
plt.imshow( np.asarray(output1) )
```




    <matplotlib.image.AxesImage at 0x2138faf6b70>




![png](output_32_1.png)



```python
plt.imshow( np.asarray(output2) )
```




    <matplotlib.image.AxesImage at 0x2138f780f98>




![png](output_33_1.png)



```python
plt.imshow( np.asarray(output32) )
```




    <matplotlib.image.AxesImage at 0x2138fb9d6a0>




![png](output_34_1.png)


### Couche 2 : Pooling 
__Question__ : 
* Que fait cette couche?
* Quel est l'intérêt de cette étape?



```python
intermediate_output_2.shape
```




    (1, 14, 14, 32)




```python
output1 = intermediate_output_2[0,:,:,0]
output2 = intermediate_output_2[0,:,:,1]
output3 = intermediate_output_2[0,:,:,31]
```


```python
plt.imshow( np.asarray(output1) )
```




    <matplotlib.image.AxesImage at 0x2138fbf3e10>




![png](output_38_1.png)



```python
plt.imshow( np.asarray(output2) )
```




    <matplotlib.image.AxesImage at 0x2138fc54c50>




![png](output_39_1.png)



```python
plt.imshow( np.asarray(output32) )
```




    <matplotlib.image.AxesImage at 0x2138fcb3a58>




![png](output_40_1.png)


### Couche 3 : 2éme Conv2D 


```python
intermediate_output_3.shape
```




    (1, 14, 14, 64)




```python
output1 = intermediate_output_3[0,:,:,0]
output2 = intermediate_output_3[0,:,:,1]
output3 = intermediate_output_3[0,:,:,63]
```


```python
plt.imshow( np.asarray(output1) )
```




    <matplotlib.image.AxesImage at 0x2138fd05278>




![png](output_44_1.png)



```python
plt.imshow( np.asarray(output2) )
```




    <matplotlib.image.AxesImage at 0x2138fd516d8>




![png](output_45_1.png)



```python
plt.imshow( np.asarray(output3) )
```




    <matplotlib.image.AxesImage at 0x2138fe64240>




![png](output_46_1.png)


### Couche 4 : Pooling


```python
intermediate_output_4.shape
```




    (1, 7, 7, 64)




```python
output1 = intermediate_output_4[0,:,:,0]
output2 = intermediate_output_4[0,:,:,1]
output3 = intermediate_output_4[0,:,:,63]
```


```python
plt.imshow( np.asarray(output1) )
```




    <matplotlib.image.AxesImage at 0x2138ff08d68>




![png](output_50_1.png)



```python
plt.imshow( np.asarray(output2) )
```




    <matplotlib.image.AxesImage at 0x2138fea7550>




![png](output_51_1.png)



```python
plt.imshow( np.asarray(output3) )
```




    <matplotlib.image.AxesImage at 0x2138ff9b438>




![png](output_52_1.png)


### Couche 5 Flatten 
__Question__ 
* Que fait cette couche ? 



```python
intermediate_output_5.shape
```




    (1, 3136)




```python
intermediate_output_5
```




    array([[0.0000000e+00, 3.2772776e-05, 0.0000000e+00, ..., 1.4001080e-02,
            7.3955031e-03, 0.0000000e+00]], dtype=float32)




```python
plt.imshow( (intermediate_output_5) )
```




    <matplotlib.image.AxesImage at 0x213902c7588>




![png](output_56_1.png)


### Compile


```python
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
```


```python

```

### TensorBoard Logger


```python
RUN_NAME = 'Model_1_Epoch_1'
import keras.callbacks as callbacks

logger = callbacks.TensorBoard(log_dir='./logs/' + RUN_NAME.format(),
    histogram_freq=5,
    write_graph=True)
```

### Train the model


```python
model.fit( x=X_train, y=y_train, batch_size=32, epochs=1, callbacks=[logger], shuffle=True, validation_data=(X_test, y_test))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/1
    60000/60000 [==============================] - 231s 4ms/step - loss: 0.0046 - acc: 0.9989 - val_loss: 0.0086 - val_acc: 0.9983
    




    <keras.callbacks.History at 0x213a16b1cc0>




```python
model.fit()
```


```python

```


```python
model.add(Conv2D(32, kernel_size=(5,5),input_shape=(28,28,1), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(5,5),padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.add(Dr)
```


```python

```


```python

```


```python

```


```python

```


```python
#plt.plot(history_cnn.history['acc'])
#plt.plot(history_cnn.history['val_acc'])
```


```python

```


```python

```

## The accuracy of the model


```python


history_cnn = cnn.fit(X_train,y_train,epochs=5,validation_data=(X_train,y_train))

plt.plot(history_cnn.history['acc'])
plt.plot(history_cnn.history['val_acc'])
```


```python
## evaluation
```
score = cnn.evaluate(X_test, y_test, batch_size=128)

```python

```


```python
#### Version 
cnn2 = Sequential()
cnn2.add(Conv2D(32, kernel_size=(3,3),input_shape=(28,28,1), padding='same', activation='relu'))
cnn2.add(MaxPooling2D())
cnn2.add(Conv2D(64, kernel_size=(3,3),padding='same', activation='relu'))
cnn2.add(MaxPooling2D())
cnn2.add(Flatten())
cnn2.add(Dense(10,activation='softmax'))
cnn2.summary()
```


```python
cnn2.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
```


```python
score = cnn2.evaluate(X_test, y_test, batch_size=128)
```


```python
#### Version SIMPLE
model_log_reg = Sequential()
model_log_reg.add( Flatten(input_shape=(28,28,1)) )
model_log_reg.add( Dense(10, activation='softmax' ) )
# this is a logistic regression in Keras
model_log_reg.summary()
model_log_reg.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
score = model_log_reg.evaluate(X_test, y_test)
```


```python
score
```
