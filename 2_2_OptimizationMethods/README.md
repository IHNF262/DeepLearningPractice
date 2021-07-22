# Optimization Methods & Model with different optimization algorithms
Request:
  python 3.8
  
In this section, we still use the typical problem of 2D points classification to illustrate the effect of different optimization methods on the speed of the parameters convergence.

__Optimizations methods__(all on the mini_batch optimizations):  
  - (1): Gradient Descent
  - (2): Model with momentum 
  - (3): Model with Adam

## Data Classification
  The NN structure & the input data
   ![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/5_1_BuildingARecurrentNeuralNetwork-tf/images/1.png)
   
   
   __Notice__: All gradient descent optimization methods are executed under the preconditionson on the mini_batch optimization.  
   The mini_batch optimization will shuffle the input X and randomly divide X into smaller batches of the same size at each iterations. 
   
   
  - __Gradient Descent__ : 


  - __Model With Momentum Optimization__ :  
     Because this example is relatively simple, the gains from using momentum are small.But for more complex problems, we will get bigger gains.


  - __Model With Adam Optimization__ :  
     Looking at the figure blow, we see that the adam optimization accelerates the parameteres convergency extremely.
  
 
  
  
