Request:
  python 3.8
  
This is a demo for planar Data Classification.
The figure on the bottom left is out input data, we except to find a non-linear function to classify these points. Obviously, the simple logestic regression can't solve this problem.
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_3_PlanarDataClassificationWithOneHiddenLayer/images/1_3_1.png)

Fundamental:
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_3_PlanarDataClassificationWithOneHiddenLayer/images/f1.png)
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_3_PlanarDataClassificationWithOneHiddenLayer/images/f2.png)
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_3_PlanarDataClassificationWithOneHiddenLayer/images/f3.png)


The classfication results based on the hidden layers containing different number of units.All results are trained through 10000 iterations.
![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_3_PlanarDataClassificationWithOneHiddenLayer/images//1_3_2.png)

The cost of this network:

![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_3_PlanarDataClassificationWithOneHiddenLayer/images/1_3_3.png)

![Alt text](https://raw.githubusercontent.com/IHNF262/DeepLearningPractice/main/1_3_PlanarDataClassificationWithOneHiddenLayer/images/1_3_4.png)
