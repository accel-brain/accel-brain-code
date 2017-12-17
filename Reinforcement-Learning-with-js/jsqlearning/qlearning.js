/*
Repository name: accel-brain-code
Description: Base class and `Template Method Pattern` of Q-Learning.
Version: 1.0.1
Author: chimera0(RUM)
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0
Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)
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
     * R(state).
     */
    var r_dict_ = {};

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
        this.strategy_ = strategy;
        if (params != undefined)
        {
            if ("alpha_value" in params)
            {
                this.alpha_value = params.alpha_value;
            }
            if ("gamma_value" in params)
            {
                this.gamma_value = params.gamma_value;
            }
            if ("q_dict" in params)
            {
                this.q_dict = params.q_dict;
            }
        }
    }

    /** @constructor */
    constructor.prototype = {
        alpha_value: alpha_value_,
        gamma_value: gamma_value_,
        q_dict: q_dict_,
        r_dict: r_dict_,

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
            try
            {
                if (this.q_dict == undefined)
                {
                    this.save_q_dict(state_key, action_key, 0.0);
                    return 0.0;
                }
                else
                {
                    var state_key_list = Object.keys(this.q_dict);
                    if (state_key in this.q_dict && this.q_dict[state_key] != undefined)
                    {
                        if (action_key in this.q_dict[state_key])
                        {
                            return this.q_dict[state_key][action_key];
                        }
                        else
                        {
                            this.save_q_dict(state_key, action_key, 0.0);
                            return 0.0;
                        }
                    }
                    else
                    {
                        this.save_q_dict(state_key, action_key, 0.0);
                        return 0.0;
                    }
                }
            }
            catch (e)
            {
                console.log(e);
                return 0.0;
            }
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
            if (q_value.isNaN == true || (q_value !== q_value) === true) return;
            if (state_key == false) return;
            if (action_key == false) return;
            if (q_value == false) return;
            if (this.q_dict == undefined) this.q_dict = {};
            if (this.q_dict[state_key] == undefined)
            {
                this.q_dict[state_key] = {};
            }
            this.q_dict[state_key][action_key] = q_value;
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
            try
            {
                if (this.r_dict == undefined)
                {
                    this.save_r_dict(state_key, 0.0, action_key);
                    return 0.0;
                }
                else
                {
                    if (state_key in this.r_dict && this.r_dict[state_key] != undefined)
                    {
                        if (action_key == undefined)
                        {
                            reward_value = this.r_dict[state_key];
                        }
                        else
                        {
                            if (action_key in this.r_dict[state_key])
                            {
                                reward_value = this.r_dict[state_key][action_key];
                            }
                            else
                            {
                                this.save_r_dict(state_key, 0.0, action_key);
                                return 0.0;
                            }
                        }
                    }
                    else
                    {
                        this.save_r_dict(state_key, 0.0, action_key);
                        return 0.0;
                    }
                }
            }
            catch(e)
            {
                console.log(e);
                return 0.0;
            }
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
            if (reward_value.isNaN == true || (reward_value !== reward_value) === true) return;
            if (state_key == false) return;
            if (reward_value == false) return;

            if (this.r_dict == undefined) this.r_dict = {};
            if (this.r_dict[state_key] == undefined)
            {
                if (action_key != undefined)
                {
                    this.r_dict[state_key] = {};
                }
                else
                {
                    this.r_dict[state_key] = 0.0;
                }
            }
            if (action_key != undefined)
            {
                this.r_dict[state_key][action_key] = reward_value;
            }
            else
            {
                this.r_dict[state_key] = reward_value;
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
            this.t = 1;
            for (var _t = 1;_t<=limit;_t++)
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
                if (this.check_the_end_flag(state_key) == true)
                {
                    break;
                }

                // Max-Q-Value in next action time.
                next_next_action_list = this.extract_possible_actions(action_key)
                next_action_key = this.predict_next_action(action_key, next_next_action_list)

                next_max_q = this.extract_q_dict(action_key, next_action_key)
                q = this.extract_q_dict(state_key, action_key);

                // Update Q-Value.
                this.update_q(
                    state_key,
                    action_key,
                    reward_value,
                    next_max_q
                );

                // Episode.
                this.t = _t;

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
            var q = this.extract_q_dict(state_key, action_key);
            // Update Q-Value.
            var new_q = q + this.alpha_value * (reward_value + (this.gamma_value * next_max_q) - q);
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
            for (var i=0; i<next_action_list.length;i++)
            {
                q = this.extract_q_dict(state_key, next_action_list[i]);
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
