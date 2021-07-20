# Building A Recurrent Neural Network(RNN)

Request:
  python 3.8
  
In this section, we try to show some typical examples with RNN or LSTM. 

## Dinasaur names: (employ RNN)

**we need a name generator to generate reasonable names, which can be constructed by training some input with existing names by RNN.**  

- (1)Open Training data :  
  There are 1536 reasonable names, and these names have a total of 19909 characters, 27 of which 27 are unique.'a'-'z' + '\n')  
  
- (2)training :  
   As the parameters, such as ___Wax, Waa, Wya, ba, by___ are unique, so the number of RNN units can vary with the length of input name characters.And in backward propagation for RNN, bias for each character all influence the gradient of parameters.

__1.Initialize parameters:__  
>(1)initialization with zeros.  
>(2)initialization with random.  
>(3)initialization with He.

__2.Model with regularization:__  
>(1)L2 regularization.  
>(2)drop-out regularization.  

-----------------------------------------------------------
1.1 Initialization with zeros:  
  we use the binary classification of 2D points to illustarte the impact of different initialization parameters on the classification results.  
  __Abviously, all points are classified into one category under the condition of initializing parameters to zero.__
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/2_1_ImprovingDeepNN_HyperparameterTuning_Regularization_Optimization/images/result/1.png)

1.2 Initialization with random:

![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/2_1_ImprovingDeepNN_HyperparameterTuning_Regularization_Optimization/images/result/2.png)

1.3 Initialization with He:

![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/2_1_ImprovingDeepNN_HyperparameterTuning_Regularization_Optimization/images/result/3.png)

2. Model with regularization:

![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/2_1_ImprovingDeepNN_HyperparameterTuning_Regularization_Optimization/images/result/4.png)
