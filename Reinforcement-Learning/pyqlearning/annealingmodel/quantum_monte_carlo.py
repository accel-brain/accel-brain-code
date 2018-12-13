# -*- coding: utf-8 -*-
import numpy as np
from pyqlearning.annealing_model import AnnealingModel
from pyqlearning.annealingmodel.distance_computable import DistanceComputable


class QuantumMonteCarlo(AnnealingModel):
    '''
    Quantum Monte Carlo.
    
    Generally, Quantum Monte Carlo is a stochastic method
    to solve the Schrödinger equation. This algorithm is one of 
    the earliest types of solution in order to simulate the Quantum Annealing 
    in classical computer. In summary, one of the function of this algorithm 
    is to solve the ground state search problem which is known as logically 
    equivalent to combinatorial optimization problem.

    References:
        - Das, A., & Chakrabarti, B. K. (Eds.). (2005). Quantum annealing and related optimization methods (Vol. 679). Springer Science & Business Media.
        - Facchi, P., & Pascazio, S. (2008). Quantum Zeno dynamics: mathematical and physical aspects. Journal of Physics A: Mathematical and Theoretical, 41(49), 493001.
        - Heim, B., Rønnow, T. F., Isakov, S. V., & Troyer, M. (2015). Quantum versus classical annealing of Ising spin glasses. Science, 348(6231), 215-217.
        - Heisenberg, W. (1925) Über quantentheoretische Umdeutung kinematischer und mechanischer Beziehungen. Z. Phys. 33, pp.879—893.
        - Heisenberg, W. (1927). Über den anschaulichen Inhalt der quantentheoretischen Kinematik und Mechanik. Zeitschrift fur Physik, 43, 172-198.
        - Heisenberg, W. (1984). The development of quantum mechanics. In Scientific Review Papers, Talks, and Books -Wissenschaftliche Übersichtsartikel, Vorträge und Bücher (pp. 226-237). Springer Berlin Heidelberg. Hilgevoord, Jan and Uffink, Jos, “The Uncertainty Principle”, The Stanford Encyclopedia of Philosophy (Winter 2016 Edition), Edward N. Zalta (ed.), URL = ＜https://plato.stanford.edu/archives/win2016/entries/qt-uncertainty/＞.
        - Jarzynski, C. (1997). Nonequilibrium equality for free energy differences. Physical Review Letters, 78(14), 2690.
        - Messiah, A. (1966). Quantum mechanics. 2 (1966). North-Holland Publishing Company.
        - Schrödinger, E. (1926). Quantisierung als eigenwertproblem. Annalen der physik, 385(13), S.437-490.
        - Somma, R. D., Batista, C. D., & Ortiz, G. (2007). Quantum approach to classical statistical mechanics. Physical review letters, 99(3), 030603.
        - 鈴木正. (2008). 「組み合わせ最適化問題と量子アニーリング: 量子断熱発展の理論と性能評価」.,『物性研究』, 90(4): pp598-676. 参照箇所はpp619-624.
        - 西森秀稔、大関真之(2018) 『量子アニーリングの基礎』須藤 彰三、岡 真 監修、共立出版、参照箇所はpp9-46.

    '''
    
    __spin_arr = None
    
    # User function for optimization.
    __distance_computable = None
    
    def __init__(
        self,
        distance_computable,
        cycles_num=100,
        inverse_temperature_beta=50,
        gammma=1.0,
        fractional_reduction=0.99,
        trotter_dimention=None,
        mc_step=None,
        point_num=None,
        spin_arr=None,
        tolerance_diff_e=None
    ):
        '''
        Init.
        
        Args:
            cycles_num:                  The number of annealing cycles.
            inverse_temperature_beta:    Inverse temperature (Beta).
            gammma:                      Gamma.
            fractional_reduction:        Attenuation rate.
            trotter_dimention:           The dimention of Trotter.
            mc_step:                     The number of Monte Carlo steps.
            point_num:                   The number of parameters.
            spin_arr:                    `np.ndarray` of 2-D spin glass in Ising model.
            tolerance_diff_e:            Tolerance for the optimization.
                                         When the ΔE is not improving by at least `tolerance_diff_e`
                                         for two consecutive iterations, annealing will stops.

        '''
        if isinstance(distance_computable, DistanceComputable):
            self.__distance_computable = distance_computable
        else:
            raise TypeError()

        self.__cycles_num = cycles_num
        self.__inverse_temperature_beta = inverse_temperature_beta
        self.__gammma = gammma
        self.__fractional_reduction = fractional_reduction
        self.__tolerance_diff_e = tolerance_diff_e
        
        if spin_arr is not None:
            if isinstance(spin_arr, np.ndarray):
                self.__spin_arr = spin_arr
            else:
                raise TypeError()

            self.__trotter_dimention = self.__spin_arr.shape[0]
            self.__mc_step = self.__spin_arr.shape[1]
            self.__point_num = self.__spin_arr.shape[2]
        else:
            if isinstance(trotter_dimention, int):
                self.__trotter_dimention = trotter_dimention
            else:
                raise TypeError()
            if isinstance(mc_step, int):
                self.__mc_step = mc_step
            else:
                raise TypeError()
            if isinstance(point_num, int):
                self.__point_num = point_num
            else:
                raise TypeError()

            arr = np.diag([1] * self.__point_num)
            arr[arr == 0] = -1
            key_arr = np.arange(self.__mc_step) % self.__point_num
            np.random.shuffle(key_arr)
            spin_arr_list = [None] * self.__trotter_dimention
            for i in range(self.__trotter_dimention):
                spin_arr_list[i] = arr[key_arr]
            self.__spin_arr = np.array(spin_arr_list)

    def annealing(self):
        '''
        Annealing.
        '''
        self.__predicted_log_list = []
        for cycle in range(self.__cycles_num):
            for mc_step in range(self.__mc_step):
                self.__move()
            self.__gammma *= self.__fractional_reduction

            if isinstance(self.__tolerance_diff_e, float) and len(self.__predicted_log_list) > 1:
                diff = abs(self.__predicted_log_list[-1][5] - self.__predicted_log_list[-2][5])
                if diff < self.__tolerance_diff_e:
                    break

        self.predicted_log_arr = np.array(self.__predicted_log_list)

    def __move(self):
        # Choice torotter.
        torotter = np.random.randint(self.__trotter_dimention)
        
        # Choice times.
        time_arr = np.arange(self.__mc_step)
        np.random.shuffle(time_arr)
        pre_time, post_time = (time_arr[0], time_arr[1])
        
        # Decide point.
        pre_point = self.__spin_arr[torotter, pre_time].argmax()
        post_point = self.__spin_arr[torotter, post_time].argmax()

        # ΔE
        delta_e_arr = np.array([0] * 5)
        for point in range(self.__point_num):
            dist_pre_point = self.__distance_computable.compute(pre_point, point)
            dist_post_point = self.__distance_computable.compute(post_point, point)
            delta_e_arr[0] += 2 * (-dist_pre_point * self.__spin_arr[torotter][pre_time][pre_point] - dist_post_point * self.__spin_arr[torotter][pre_time][post_point]) * (self.__spin_arr[torotter][pre_time - 1][point] + self.__spin_arr[torotter][(pre_time + 1) % self.__mc_step][point])
            delta_e_arr[0] += 2 + (-dist_pre_point * self.__spin_arr[torotter][pre_time][post_point] - dist_post_point * self.__spin_arr[torotter][post_time][post_point]) * (self.__spin_arr[torotter][post_time - 1][point] + self.__spin_arr[torotter][(post_time + 1) % self.__mc_step][point])

        annealing_e = (1 / self.__inverse_temperature_beta) * np.log(np.cosh(self.__inverse_temperature_beta * self.__gammma / self.__trotter_dimention) / np.sinh(self.__inverse_temperature_beta * self.__gammma / self.__trotter_dimention))

        delta_e_arr[1] = self.__spin_arr[torotter][pre_time][pre_point] * (self.__spin_arr[(torotter - 1) % self.__trotter_dimention][pre_time][pre_point] + self.__spin_arr[(torotter + 1) % self.__trotter_dimention][pre_time][pre_point])
        delta_e_arr[2] = self.__spin_arr[torotter][pre_time][post_time] * (self.__spin_arr[(torotter - 1) % self.__trotter_dimention][pre_time][post_point] + self.__spin_arr[(torotter + 1) % self.__trotter_dimention][pre_time][post_point])
        delta_e_arr[3] = self.__spin_arr[torotter][post_time][pre_point] * (self.__spin_arr[(torotter - 1) % self.__trotter_dimention][post_time][pre_point] + self.__spin_arr[(torotter + 1) % self.__trotter_dimention][post_time][pre_point])
        delta_e_arr[4] = self.__spin_arr[torotter][post_time][post_point] * (self.__spin_arr[(torotter - 1) % self.__trotter_dimention][post_time][post_point] + self.__spin_arr[(torotter + 1) % self.__trotter_dimention][post_time][post_point])

        delta_e = delta_e_arr[0] / self.__trotter_dimention + annealing_e * delta_e_arr[1:].sum()

        # Flip or not.
        flip_flag = False
        prob = 0.0
        if delta_e <= 0:
            flip_flag = True
        else:
            prob = np.exp(-self.__inverse_temperature_beta * self.__gammma)
            if np.random.binomial(1, prob) == 1:
                flip_flag = True

        if flip_flag is True:
            self.__spin_arr[torotter][pre_time][pre_point] *= -1
            self.__spin_arr[torotter][pre_time][post_point] *= -1
            self.__spin_arr[torotter][post_time][pre_point] *= -1
            self.__spin_arr[torotter][post_time][post_point] *= -1

        self.__predicted_log_list.append(
            (
                torotter,
                pre_time,
                post_time,
                pre_point,
                post_point,
                delta_e,
                prob,
                flip_flag
            )
        )

    def get_spin_arr(self):
        ''' getter '''
        return self.__spin_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    spin_arr = property(get_spin_arr, set_readonly)
