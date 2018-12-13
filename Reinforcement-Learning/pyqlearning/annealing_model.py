# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import warnings


class AnnealingModel(metaclass=ABCMeta):
    '''
    Abstract class of Annealing.
    
    There are many hyperparameters that we have to set before 
    the actual searching and learning process begins. 
    Each parameter should be decided in relation to Deep/Reinforcement Learning theory
    and it cause side effects in training model. Because of this complexity of hyperparameters, 
    so-called the hyperparameter tuning must become a burden of Data scientists 
    and R & D engineers from the perspective of not only a theoretical point of view 
    but also implementation level.

    This issue can be considered as Combinatorial optimization problem which is 
    an optimization problem, where an optimal solution has to be identified from 
    a finite set of solutions. The solutions are normally discrete or can be converted 
    into discrete. This is an important topic studied in operations research such as 
    software engineering, artificial intelligence(AI), and machine learning. For instance, 
    travelling sales man problem is one of the popular combinatorial optimization problem.

    In this problem setting, this library provides an Annealing Model to search optimal 
    combination of hyperparameters. For instance, Simulated Annealing is a probabilistic
    single solution based search method inspired by the annealing process in metallurgy.
    Annealing is a physical process referred to as tempering certain alloys of metal, 
    glass, or crystal by heating above its melting point, holding its temperature, 
    and then cooling it very slowly until it solidifies into a perfect crystalline structure. 
    The simulation of this process is known as simulated annealing.
    
    References:
        - Bektas, T. (2006). The multiple traveling salesman problem: an overview of formulations and solution procedures. Omega, 34(3), 209-219.
        - Bertsimas, D., & Tsitsiklis, J. (1993). Simulated annealing. Statistical science, 8(1), 10-15.
        - Das, A., & Chakrabarti, B. K. (Eds.). (2005). Quantum annealing and related optimization methods (Vol. 679). Springer Science & Business Media.
        - Du, K. L., & Swamy, M. N. S. (2016). Search and optimization by metaheuristics. New York City: Springer.
        - Edwards, S. F., & Anderson, P. W. (1975). Theory of spin glasses. Journal of Physics F: Metal Physics, 5(5), 965.
        - Facchi, P., & Pascazio, S. (2008). Quantum Zeno dynamics: mathematical and physical aspects. Journal of Physics A: Mathematical and Theoretical, 41(49), 493001.
        - Heim, B., Rønnow, T. F., Isakov, S. V., & Troyer, M. (2015). Quantum versus classical annealing of Ising spin glasses. Science, 348(6231), 215-217.
        - Heisenberg, W. (1925) Über quantentheoretische Umdeutung kinematischer und mechanischer Beziehungen. Z. Phys. 33, pp.879—893.
        - Heisenberg, W. (1927). Über den anschaulichen Inhalt der quantentheoretischen Kinematik und Mechanik. Zeitschrift fur Physik, 43, 172-198.
        - Heisenberg, W. (1984). The development of quantum mechanics. In Scientific Review Papers, Talks, and Books -Wissenschaftliche Übersichtsartikel, Vorträge und Bücher (pp. 226-237). Springer Berlin Heidelberg. Hilgevoord, Jan and Uffink, Jos, “The Uncertainty Principle”, The Stanford Encyclopedia of Philosophy (Winter 2016 Edition), Edward N. Zalta (ed.), URL = ＜https://plato.stanford.edu/archives/win2016/entries/qt-uncertainty/＞.
        - Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. Physical Review Letters, 78(14), 2690.
        - Messiah, A. (1966). Quantum mechanics. 2 (1966). North-Holland Publishing Company.
        - Mezard, M., & Montanari, A. (2009). Information, physics, and computation. Oxford University Press.
        - Nallusamy, R., Duraiswamy, K., Dhanalaksmi, R., & Parthiban, P. (2009). Optimization of non-linear multiple traveling salesman problem using k-means clustering, shrink wrap algorithm and meta-heuristics. International Journal of Nonlinear Science, 8(4), 480-487.
        - Schrödinger, E. (1926). Quantisierung als eigenwertproblem. Annalen der physik, 385(13), S.437-490.
        - Somma, R. D., Batista, C. D., & Ortiz, G. (2007). Quantum approach to classical statistical mechanics. Physical review letters, 99(3), 030603.
        - 鈴木正. (2008). 「組み合わせ最適化問題と量子アニーリング: 量子断熱発展の理論と性能評価」.,『物性研究』, 90(4): pp598-676. 参照箇所はpp619-624.
        - 西森秀稔、大関真之(2018) 『量子アニーリングの基礎』須藤 彰三、岡 真 監修、共立出版、参照箇所はpp9-46.

    '''
    # The fitted data points.
    __var_arr = None

    # The np.ndarray of the log of predicted score.
    __predicted_log_arr = None

    # Optimized data points.
    __var_log_arr = None

    # The `np.ndarray` of computed cost.
    __computed_cost_arr = None

    def fit_dist_mat(self, dist_mat_arr):
        '''
        Fit ovserved data points.

        Args:
            dist_mat_arr:    fitted data points.
        '''
        warnings.warn("This property will be removed in future version. Use `var_arr`.", FutureWarning)
        # Set the data points.
        self.var_arr = dist_mat_arr
        
    @abstractmethod
    def annealing(self):
        '''
        Annealing.
        '''
        raise NotImplementedError()

    def get_var_arr(self):
        ''' getter '''
        if isinstance(self.__var_arr, np.ndarray):
            return self.__var_arr
        else:
            raise TypeError()
    
    def set_var_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__var_arr = value
        else:
            raise TypeError()

    var_arr = property(get_var_arr, set_var_arr)

    def get_predicted_log_arr(self):
        ''' getter '''
        if isinstance(self.__predicted_log_arr, np.ndarray):
            return self.__predicted_log_arr
        else:
            raise TypeError()
    
    def set_predicted_log_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__predicted_log_arr = value
        else:
            raise TypeError()
    
    predicted_log_arr = property(get_predicted_log_arr, set_predicted_log_arr)

    def get_var_log_arr(self):
        ''' getter '''
        if isinstance(self.__var_log_arr, np.ndarray):
            return self.__var_log_arr
        else:
            raise TypeError()
    
    def set_var_log_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__var_log_arr = value
        else:
            raise TypeError()

    var_log_arr = property(get_var_log_arr, set_var_log_arr)

    def get_computed_cost_arr(self):
        ''' getter '''
        if isinstance(self.__computed_cost_arr, np.ndarray):
            return self.__computed_cost_arr
        else:
            raise TypeError()
    
    def set_computed_cost_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray):
            self.__computed_cost_arr = value
        else:
            raise TypeError()
    
    computed_cost_arr = property(get_computed_cost_arr, set_computed_cost_arr)


#########################################################################################################    
# Removed in future version.
#########################################################################################################    

    def get_predicted_log_list(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `predicted_log_arr`.", FutureWarning)
        return self.predicted_log_arr

    def set_predicted_log_list(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `predicted_log_arr`.", FutureWarning)
        raise TypeError("This property must be read-only.")

    predicted_log_list = property(get_predicted_log_list, set_predicted_log_list)

    def get_x(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr`.", FutureWarning)
        return self.var_log_arr

    def set_x(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr`.", FutureWarning)
        self.var_log_arr = value

    x = property(get_x, set_x)

    def get_stocked_predicted_arr(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr`.", FutureWarning)
        return self.computed_cost_arr

    def set_stocked_predicted_arr(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr`.", FutureWarning)
        raise TypeError("This property must be read-only.")

    stocked_predicted_arr = property(get_stocked_predicted_arr, set_stocked_predicted_arr)

    def get_current_dist_arr(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr[-1]`.", FutureWarning)
        return self.var_log_arr[-1]

    def set_current_dist_arr(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `var_log_arr`.", FutureWarning)
        raise TypeError("This property must be read-only.")

    current_dist_arr = property(get_current_dist_arr, set_current_dist_arr)

    def get_current_cost_arr(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr[-1]`.", FutureWarning)
        return self.computed_cost_arr[-1]

    def set_current_cost_arr(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version. Use `computed_cost_arr[-1]`.", FutureWarning)
        raise TypeError("This property must be read-only.")
    
    current_cost_arr = property(get_current_cost_arr, set_current_cost_arr)
    
    def get_accepted_pos(self):
        ''' getter '''
        warnings.warn("This property will be removed in future version.", FutureWarning)
        return None

    def set_accepted_pos(self, value):
        ''' setter '''
        warnings.warn("This property will be removed in future version.", FutureWarning)
        raise TypeError("This property must be read-only.")

    accepted_pos = property(get_accepted_pos, set_accepted_pos)
