/*
Repository name: accel-brain-code
Description: Base class and `Template Method Pattern` of Q-Learning.
Version: 1.0.1
Author: chimera0(RUM)
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0
Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)
*/

/**
 * Base class and `Template Method Pattern` of Q-Learning.
 */
var QLearning = (function()
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
     * Learning rate.
     */
    var alpha_value_ = 0.1;

    /** 
     * @private
     *
     * Gamma value.
     */
    var gamma_value_ = 0.5;

    /** 
     * @private
     *
     * Q(state, action).
     */
    var q_dict_ = {};

    /** 
     * @private
     *
     * Time.
     */
    var t_ = 0;

    /**
     * Set Up hyperparams.
     *
     * @constructor
     * @params {object}
     */
    var constructor = function(strategy, params) {
        strategy_ = strategy;
        if (params != undefined)
        {
            if ("alpha_value" in params)
            {
                alpha_value_ = params.alpha_value;
            }
            if ("gamma_value" in params)
            {
                gamma_value_ = params.gammma_value;
            }
            if ("q_dict" in params)
            {
                q_dict_ = params.q_dict;
            }
        }
    }

    /** @constructor */
    constructor.prototype = {
        alpha_value: alpha_value_,
        gamma_value: gamma_value_,
        q_dict: q_dict_,

        /**
         *
         * Extract Q-Value from `self.q_dict`.
         *
         * @params {string}
         * @params {string}
         *
         * @return {float}
         */
        extract_q_dict: function (state_key, action_key)
        {
            q = q_dict_[state_key][action_key];
            if (q != undefined) return q;
            return 0.0
        },

        /**
         *
         * Insert or update Q-Value in `self.q_dict`.
         *
         * @params {string}
         * @params {string}
         * @params {float}
         */
        save_q_dict: function(state_key, action_key, q_value)
        {
            if (q_dict_[state_key] == undefined)
            {
                q_dict[state_key] = {};
            }
            q_dict_[state_key][action_key] = q_value;
        },

        /**
         *
         * Extract R-Value from `self.r_dict`.
         *
         * @params {string}
         * @params {string}
         *
         * @return {float}
         */
        extract_r_dict : function (state_key, action_key)
        {
            if (action_key == undefined)
            {
                reward_value = r_dict_[state_key];
            }
            else
            {
                reward_value = r_dict_[state_key][action_key];
            }
            if (reward_value != undefined) return reward_value;
            return 0.0;
        },

        /**
         *
         * Insert or update Reward Value in `self.r_dict`.
         *
         * @params {string}
         * @params {string}
         * @params {float}
         */
        save_r_dict: function (state_key, reward_value, action_key)
        {
            if (action_key == undefined)
            {
                r_dict_[state_key] = reward_value;
            }
            else
            {
                r_dict_[state_key][action_key] = reward_value;
            }
        },

        /**
         * Time.
         */
        t: t_,

        /**
         *
         * Learning.
         *
         * @params {string}
         * @params {int}
         */
        learn: function (state_key, limit)
        {
            if (limit == undefined) limit = 10000;
            t_ = 1;
            while (t_ <= limit)
            {
                next_action_list = this.extract_possible_actions(state_key);
                action_key = this.select_action(
                    state_key,
                    next_action_list
                )
                reward_value = this.observe_reward_value(state_key, action_key)

                // Vis.
                this.visualize_learning_result(state_key)
                // Check.
                if (self.check_the_end_flag(state_key) == true)
                {
                    break
                }

                // Max-Q-Value in next action time.
                next_next_action_list = this.extract_possible_actions(action_key)
                next_action_key = this.predict_next_action(action_key, next_next_action_list)
                next_max_q = this.extract_q_dict(action_key, next_action_key)

                // Update Q-Value.
                this.update_q(
                    state_key,
                    action_key,
                    reward_value,
                    next_max_q
                )

                // Epsode.
                self.t += 1

                // Update State.
                state_key = this.update_state(
                    state_key,
                    action_key
                )
            }
        },

        /**
         * Select action by Q(state, action).
         *
         * @params{string}
         * @params{array}
         *
         * @return{string}
         */
        select_action: function (state_key, next_action_list)
        {
            return this.strategy_.select_action(this, state_key, next_action_list);
        },

        /**
         * Extract the list of the possible action in `self.t+1`.
         *
         * @params{string}
         *
         * @return{array}
         */
        extract_possible_actions: function (state_key)
        {
            return this.strategy_.extract_possible_actions(this, state_key);
        },

        /**
         * Extract the list of the possible action in `self.t+1`.
         *
         * @params{string}
         * @params{string}
         *
         * @return{array}
         */
        observe_reward_value: function (state_key, action_key)
        {
            return this.strategy_.observe_reward_value(this, state_key, action_key);
        },

        /**
         * Update Q-Value.
         *
         * @params{string}
         * @params{string}
         * @params{float}
         * @params{float}
         *
         */
        update_q: function (state_key, action_key, reward_value, next_max_q)
        {
            // Now Q-Value.
            q = this.extract_q_dict(state_key, action_key);
            // Update Q-Value.
            new_q = q + this.alpha_value * (reward_value + (this.gamma_value * next_max_q) - q);
            // Save updated Q-Value.
            this.save_q_dict(state_key, action_key, new_q);
        },

        /**
        * Predict next action by Q-Learning.
        *
        * @params{string}
        * @params{array}
        *
        * @return{string}
        */
        predict_next_action(state_key, next_action_list)
        {
            next_action_q_list = [];
            max_q = 0.0;
            max_q_action = null;
            for (action_key in next_action_list)
            {
                q = this.extract_q_dict(state_key, action_key);
                if (max_q < q)
                {
                    max_q = q;
                    max_q_action = action_key;
                }
            }
            return max_q_action;
        },

        /**
         * Update state.
         *
         * @params{string}
         * @params{string}
         *
         * @return{string}
         */
        update_state: function(state_key, action_key)
        {
            if (this.strategy_.update_state == undefined)
            {
                return action_key;
            }
            else
            {
                return this.strategy_.update_state(this, state_key, action_key);
            }
        },
        
        /**
         * Check the end flag.
         *
         * If this return value is `True`, the learning is end.
         * As a rule, the learning can not be stopped.
         * This method should be overrided for concreate usecases.
         *
         * @params{string}
         *
         * @return{bool}
         */
        check_the_end_flag: function(state_key)
        {
            if (this.strategy_.check_the_end_flag == undefined)
            {
                return false;
            }
            else
            {
                return this.check_the_end_flag(this, state_key)
            }
        },
        
        /**
         *  Visualize learning result.
         *
         * @params{string}
         *
         */
        visualize_learning_result: function (state_key)
        {
            if (this.strategy_.visualize_learning_result != undefined)
            {
                this.strategy_.visualize_learning_result(this, state_key);
            }
        }
    };
    return constructor;
}
) ();
