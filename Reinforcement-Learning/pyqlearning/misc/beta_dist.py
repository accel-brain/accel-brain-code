

class BetaDist(object):
    '''
    Beta Distribusion for Thompson Sampling.
    '''
    # Alpha
    __default_alpha = 1
    # Beta
    __default_beta = 1
    # The number of success.
    __success = 0
    # The number of failure.
    __failure = 0

    def __init__(self, default_alpha=1, default_beta=1):
        '''
        Initialization

        Args:
            default_alpha:      Alpha
            default_beta:       Beta

        '''
        if isinstance(default_alpha, int) is False:
            if isinstance(default_alpha, float) is False:
                raise TypeError()
        if isinstance(default_beta, int) is False:
            if isinstance(default_beta, float) is False:
                raise TypeError()

        if default_alpha <= 0:
            raise ValueError()
        if default_beta <= 0:
            raise ValueError()

        self.__success += 0
        self.__failure += 0
        self.__default_alpha = default_alpha
        self.__default_beta = default_beta

    def observe(self, success, failure):
        '''
        Observation data.

        Args:
            success:      The number of success.
            failure:      The number of failure.

        '''
        if isinstance(success, int) is False:
            if isinstance(success, float) is False:
                raise TypeError()
        if isinstance(failure, int) is False:
            if isinstance(failure, float) is False:
                raise TypeError()

        if success <= 0:
            raise ValueError()
        if failure <= 0:
            raise ValueError()

        self.__success += success
        self.__failure += failure

    def likelihood(self):
        '''
        Compute likelihood.

        Returns:
            likelihood.
        '''
        try:
            likelihood = self.__success / (self.__success + self.__failure)
        except ZeroDivisionError:
            likelihood = 0.0
        return likelihood

    def expected_value(self):
        '''
        Compute expected value.

        Returns:
            Expected value.
        '''
        alpha = self.__success + self.__default_alpha
        beta = self.__failure + self.__default_beta

        try:
            expected_value = alpha / (alpha + beta)
        except ZeroDivisionError:
            expected_value = 0.0
        return expected_value

    def variance(self):
        '''
        Compute variance.

        Returns:
            variance.
        '''
        alpha = self.__success + self.__default_alpha
        beta = self.__failure + self.__default_beta

        try:
            variance = alpha * beta / ((alpha + beta) ** 2) * (alpha + beta + 1)
        except ZeroDivisionError:
            variance = 0.0
        return variance
