Request:
  python 3.8
  
In this section, we try to employ the multilayer NN to classify images of cat.  

Fundamental:
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_4_BuildingYourDeepNeuralNetwork/images/1_1.png)

The costs of a two-layer NN trained for 2500 iterations. (dims_layer = [12288, 7, 1], 12288 is the flatten data of input 64x64 image)  

The accruacy of training sets is 100%.  
The accruacy of test sets is 76%.  

__Notice__: the initial parameter of W is multiplied by the sqrt(1/n_h[L-1]), n_h[L-1] is the number of units in last layer.
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_4_BuildingYourDeepNeuralNetwork/images/1.png)
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_4_BuildingYourDeepNeuralNetwork/images/2.png)




The costs of a four-layer NN trained for 2500 iterations. (dims_layer = [12288, 20, 7, 5, 1])
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_4_BuildingYourDeepNeuralNetwork/images/3.png)
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_4_BuildingYourDeepNeuralNetwork/images/4.png)

The wrong results classified by this four-layer NN:
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_4_BuildingYourDeepNeuralNetwork/images/5.png)

