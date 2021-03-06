# imitation_learning
This is the GAIL code part for ECE239 course project "Study of the Generative Adversarial ImitationLearning Algorithm"

## Requirement
To run the code, please first install the stable baseline package. Please follow the instructions on this link [https://github.com/hill-a/stable-baselines]

## Run each experiment
In our project, we do experiments on CartPole-v1, Pendulum-v0, LunarLander Continuous-v2

### PPO + GAIL

#### CartPole-v1
<div align="center">
  <img src = './images/CartPole.png' width = '200px' height = '200px'>
</div>
CartPole-v1 is an easy task, we train our expert model for 1 time with the default parameter setting here. To reproduce our result, run:


```
> python PPO_CartPole.py
```

#### Pendulum-v0
<div align="center">
  <img src = './images/Pendulum.png' width = '200px' height = '200px'>
</div>
Based on our experiences, Pendulum-v0 is harder than CartPole-v1. Thus we train the expert model for 3 times and pick the best one as our final model. We use multiprocessing model and environment to facilitate training the expert model. To reproduce our results, run:


```
> python PPO_multienv_pendulum.py
```

#### LunarLander Continuous-v2
<div align="center">
  <img src = './images/LunarLander_continuous.png' width = '200px' height = '200px'>
</div>
For LunarLander Continous-v2, we also train the expert model for 3 times and pick the best one as our final model. Again, we use multiprocessing model and environment to facilitate training the expert model. To reproduce our results, run:


```
> python PPO_multienv_Lunar.py
```


### SAC + GAIL
For SAC + GAIL, we put our colab code for LunarLander Continous-v2 here. Please see the colab file SAC_GAIL.ipynb 

