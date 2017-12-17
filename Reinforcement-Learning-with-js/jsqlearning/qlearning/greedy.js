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
 * Q-Learning with Epsilon-Greedy.
 *
 */
var Greedy = (function()
{
    /**
     *
     * @private
     *
     */
    var strategy_ = null;

    /**
     *
     * @private
     *
     */
    var epsilon_greedy_rate_ = 0.75;

    /**
     * Set Up hyperparams.
     *
     * @constructor
     * @params {object}
     */
    var constructor = function(strategy, params) {
        this.strategy_ = strategy;
        if ("epsilon_greedy_rate" in params)
        {
            this.epsilon_greedy_rate = params.epsilon_greedy_rate;
        }
    }

    /** @constructor */
    constructor.prototype = {
        epsilon_greedy_rate: epsilon_greedy_rate_,
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
            var greedy_prob = Math.random();
            if (greedy_prob > 1.0 - this.epsilon_greedy_rate)
            {
                var action_key = __self__.predict_next_action(state_key, next_action_list);
            }
            else
            {
                var min = 0;
                var max = next_action_list.length;
                var key = Math.floor(Math.random() * (max + 1 - min) ) + min;
                var action_key = next_action_list[key];
            }
            return action_key;
        }
    };
    return constructor;

}) ();
