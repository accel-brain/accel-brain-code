/*
Repository name: accel-brain-code
Description: base class and `Template Method Pattern` of Q-Learning.
Version: 1.0.1
Author: chimera0(RUM)
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0
Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)
*/


/**
 * Q-Learning with Boltzmann distribution.
 *
 */
var Boltzmann = (function()
{
    /**
     *
     * @private
     *
     */
    var strategy_ = null;

    /** 
     * @private
     *
     * Time rate.
     */
    var time_rate_ = 0.001;

    /**
     * Set Up hyperparams.
     *
     * @constructor
     * @params {object}
     */
    var constructor = function(strategy, params) {
        this.strategy_ = strategy;
        if ("time_rate" in params)
        {
            this.time_rate = params.time_rate;
        }
    }

    /** @constructor */
    constructor.prototype = {
        time_rate: time_rate_,
        /**
         * Select action by Q(state, action).
         * Concreat method for boltzmann distribution.
         *
         * @params{string}
         * @params{array}
         *
         * @return{string}
         */
        select_action: function (__self__, state_key, next_action_list)
        {
            var next_action_b_list = calculate_boltzmann_factor_(__self__, state_key, next_action_list);
            if (next_action_b_list.length == 0)
            {
                if (next_action_list.length == 0)
                {
                    return false;
                }
                else
                {
                    var _next_action_list = next_action_list.filter(function (x, i, self)
                    {
                        return self.indexOf(x) === i;
                    });
                    return _next_action_list[Math.floor(Math.random() * _next_action_list.length)];
                }
            }
            if (next_action_b_list.length == 1)
            {
                return next_action_b_list[0][0]
            }
            var prob = Math.random();
            var i = 0;
            while (prob > next_action_b_list[i][1] + next_action_b_list[i+1][1])
            {
                i += 1;
                if (i+1 >= next_action_b_list.length)
                {
                    break;
                }
            }
            var max_b_action_key = next_action_b_list[i][0]
            return max_b_action_key
        },
        /**
         * Extract the list of the possible action in `self.t+1`.
         *
         * @params{string}
         *
         * @return{array}
         */
        extract_possible_actions: function (__self__, state_key)
        {
            return this.strategy_.extract_possible_actions(__self__, state_key);
        },

        /**
         * Extract the list of the possible action in `self.t+1`.
         *
         * @params{string}
         * @params{string}
         *
         * @return{array}
         */
        observe_reward_value: function (__self__, state_key, action_key)
        {
            return this.strategy_.observe_reward_value(__self__, state_key, action_key);
        }
    }

    /**
     * Function of temperature.
     *
     * @return{float}
     */
    var calculate_sigmoid_ = function (__self__)
    {
        var sigmoid = 1 / Math.log(__self__.t * this.time_rate + 1.1)
        return sigmoid;
    }

    /**
     * Calculate boltzmann factor.
     *
     * @params{string}
     * @params{array}
     *
     * @return{array}
     */
    var calculate_boltzmann_factor_ = function(__self__, state_key, next_action_list)
    {
        var sigmoid = calculate_sigmoid_(__self__);
        var parent_list = [];
        var action_key_list = [];
        var parent_sum = 0;
        for (var i = 0;i<next_action_list.length;i++)
        {
            var action_key = next_action_list[i];
            parent = Math.exp(__self__.extract_q_dict(state_key, action_key)) / sigmoid;
            parent_list.push(parent);
            action_key_list.push(action_key);
            parent_sum += parent;
        }
        var next_action_b_list = [];
        for (var i = 0; i<parent_list.length;i++)
        {
            var child_b = parent_list[i] / parent_sum;
            next_action_b_list.push([action_key_list[i], child_b]);
        }
        return next_action_b_list;
    }
    return constructor;

}) ();
