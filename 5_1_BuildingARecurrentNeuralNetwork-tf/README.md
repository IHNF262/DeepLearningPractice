Request:
  python 3.8
  
In this section, we try to show some typical examples with RNN or LSTM.

__1.Dinasaur names



__1

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
