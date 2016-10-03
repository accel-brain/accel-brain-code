#!/user/bin/env python
# -*- coding: utf-8 -*-


class Normalization(object):
    '''
    規格化定数メソッド
    '''

    def z_theta(self, target_list):
        '''
        1 / z(θ)
        和が1になるように整形

        Args:
            int or floatのリスト

        Returns:
            規格化したリスト
        '''
        try:
            result_list = [target / sum(target_list) for target in target_list]
        except ZeroDivisionError:
            self.pre_setup(
                x_min=min(target_list),
                x_max=max(target_list),
                range_min=0,
                range_max=1
            )
            target_list = [self.listup(target) for target in target_list]
            try:
                result_list = [target / sum(target_list) for target in target_list]
            except ZeroDivisionError:
                result_list = target_list

        return result_list

    def pre_setup(self, x_min, x_max, range_min=0, range_max=1):
        '''
        リスト内包などで一括で正規化する場合の備え

        Args:
            x_min:      最小値
            x_max:      最大値
            range_min:  規格化後の最小値
            range_max:  規格化後の最大値

        '''
        self.__x_min = x_min
        self.__x_max = x_max
        self.__range_min = range_min
        self.__range_max = range_max

    def listup(self, x):
        '''
        リスト内保などで一括で正規化する
        Args:
            x:          対象データ
        Returns:
            規格化
        '''
        return self.once(
            x=x,
            x_min=self.__x_min,
            x_max=self.__x_max,
            range_min=self.__range_min,
            range_max=self.__range_max
        )

    def once(self, x, x_min, x_max, range_min=0, range_max=1):
        '''
        規格化する
        Yn=B+(A-B)*(Xn-Xmin)/(Xmax-Xmin)

        Args:
            x:          対象データ
            x_min:      最小値
            x_max:      最大値
            range_min:  規格化後の最小値
            range_max:  規格化後の最大値

        Returns:
            規格化
        '''
        try:
            result = range_min + (range_max - range_min) * (x - x_min) / (x_max - x_min)
        except ZeroDivisionError:
            result = 0.0
        finally:
            return result
