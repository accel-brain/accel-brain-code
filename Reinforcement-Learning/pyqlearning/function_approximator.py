# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class FunctionApproximator(metaclass=ABCMeta):
    '''
    The interface of Function Approximators.
    
    Typically, the Deep Q-Learning such as the Deep Q-Network uses the 
    Convolutional Neural Networks(CNN) as a function approximator
    to solve problem setting of so-called Combination explosion.
    
    But it is not inevitable to functionally reuse CNN as 
    a function approximator. In the above problem setting of 
    generalisation and Combination explosion, for instance, 
    Long Short-Term Memory(LSTM) networks, which is-a special 
    Reccurent Neural Network(RNN) structure, and CNN as a function 
    approximator are functionally equivalent. In the same problem 
    setting, functional equivalents can be functionally replaced. 
    Considering that the feature space of the rewards has the 
    time-series nature, LSTM will be more useful.
    
    This interface defines methods to controll functionally equivalents
    of CNN. `DeepQLearning` can be delegated an object that is-a this interface.
    More detail, this interface defines a family of algorithms of Deep Learning,
    such as LSTM, Convolutional LSTM(Xingjian, S. H. I. et al., 2015), and 
    CLDNN Architecture(Sainath, T. N, et al., 2015) encapsulate each one, 
    and make them interchangeable.  Strategy lets the function approximation 
    algorithm vary independently from the clients that use it. 
    Capture the abstraction in an interface, bury implementation details in derived classes.

    References:
        - https://code.accel-brain.com/Deep-Learning-by-means-of-Design-Pattern/README.html
        - https://code.accel-brain.com/Reinforcement-Learning/README.html#deep-q-network
        - [Egorov, M. (2016). Multi-agent deep reinforcement learning.](https://pdfs.semanticscholar.org/dd98/9d94613f439c05725bad958929357e365084.pdf)
        - Gupta, J. K., Egorov, M., & Kochenderfer, M. (2017, May). Cooperative multi-agent control using deep reinforcement learning. In International Conference on Autonomous Agents and Multiagent Systems (pp. 66-83). Springer, Cham.
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
        - Sainath, T. N., Vinyals, O., Senior, A., & Sak, H. (2015, April). Convolutional, long short-term memory, fully connected deep neural networks. In Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on (pp. 4580-4584). IEEE.
        - Xingjian, S. H. I., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. In Advances in neural information processing systems (pp. 802-810).

    '''

    @abstractproperty
    def model(self):
        '''
        `object` of model as a function approximator.
        '''
        raise NotImplementedError("This property must be implemented.")

    @abstractmethod
    def learn_q(self, predicted_q_arr, real_q_arr):
        '''
        Infernce Q-Value.
        
        Args:
            predicted_q_arr:    `np.ndarray` of predicted Q-Values.
            real_q_arr:         `np.ndarray` of real Q-Values.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def inference_q(self, next_action_arr):
        '''
        Infernce Q-Value.
        
        Args:
            next_action_arr:     `np.ndarray` of action.
        
        Returns:
            `np.ndarray` of Q-Values.
        '''
        raise NotImplementedError("This method must be implemented.")
