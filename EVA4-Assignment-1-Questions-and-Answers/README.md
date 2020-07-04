# EVA4-Assignment-1-Questions-and-Answers



1.	Channels: It is the region which contains all the features(texture, patterns, part of objects)  of an image, needs to be extracted by feature extractor.

2.	Kernels : Kernels are the feature extractor, used to extract features of the given image.

3.	3x3 Kernels :  3x3 Kernel is a feature extractor, used to extract features of given image. This is most commonly used because of its size. It covers all the pixels without much overlapping irrespective of 5x5 and 7x7 kernels.  

 
4.	we need to perform 3x3 convolutions operations 100 times to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
eg : 199x199| 3x3| 197x197
        197x197|3x3|195x195
        195x195|3x3|193x193
        193x193|3x3| 191x191
        191x191|3x3| 189x189
           ……
        |3x3||3x3| 1x1

5.	Kernel initializer : Kernel initializer is used to initialize the weights used in convolution. It sets initial random weights of keras layers.
                       Eg: model.add(Dense(64,
                                           kernel_initializer='random_uniform',
                                           bias_initializer='zeros'))

6.	During training of DNN, the kernels extract feature like patterns, texture of an image from receptive fields. Then, the neural network is trained to learn data through forward and backward propagations for n number of epochs to achieve desired results 

