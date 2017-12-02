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
var Autocompletion = (function()
{
    /**
     *
     * @private
     *
     */
    var nlp_base_ = null;

    /**
     *
     * @private
     *
     */
    var n_gram_ = null;

    /**
     *
     * @private
     *
     */
    var n_ = 2;

    /**
     *
     * @private
     *
     */
    var state_action_list_dict_ = {}

    /**
     * Set Up hyperparams.
     *
     * @constructor
     * @params {object}
     * @params {object}
     * @params {object}
     * @params {int}
     */
    var constructor = function(nlp_base, n_gram, n) {
        this.nlp_base_ = nlp_base;
        this.n_gram_ = n_gram;
        if (n != undefined) this.n_ = n;
    }

    /** @constructor */
    constructor.prototype = {
        /**
         * Pre-training.
         *
         * @params{string}
         */
        pre_training: function (__self__, __document__)
        {
            this.nlp_base_.tokenize(__document__);
            token_list = this.nlp_base_.token;
            token_tuple_zip = this.n_gram_.generate_ngram_data_set(
                token_list,
                this.n_
            )
            for (var i=0;i<token_tuple_zip.length;i++)
            {
                var token_tuple = token_tuple_zip[i];
                setup_r_q_(__self__, token_tuple[0], token_tuple[1]);
            }
        },
        lap_extract_ngram : function(__self__, __document__)
        {
            this.nlp_base_.tokenize(__document__);
            token_list = this.nlp_base_.token;
            if (token_list.length > __self__.n)
            {
                token_tuple_zip = this.n_gram_.generate_ngram_data_set(
                    token_list,
                    __self__.n
                );
                token_tuple_list = [];
                for (var i = 0;i<token_tuple_zip.length;i++)
                {
                    token_tuple_list.push(token_tuple_zip[1]);
                }
                return token_tuple_list[token_tuple_list.length + 1];
            }
            else
            {
                return token_list[token_list.length-1];
            }
        },
        extract_possible_actions: function(__self__, state_key)
        {
            if (this.state_action_list_dict != undefined && state_key in this.state_action_list_dict)
            {
                return this.state_action_list_dict[state_key];
            }
            else
            {
                action_list = [];
                var state_key_list = Object.keys(this.state_action_list_dict);
                for (var i = 0; i<state_key_list.length;i++)
                {
                    if (state_key_list[i].indexOf(state_key) != -1)
                    {
                        action_list.push(this.state_action_list_dict[state_key_list[i]]);
                    }
                }
                return action_list;
            }
        },
        observe_reward_value: function(__self__, state_key, action_key)
        {
            reward_value = 0.0
            if (state_key in this.state_action_list_dict)
            {
                if (this.state_action_list_dict[state_key].indexOf(action_key) != -1)
                {
                    reward_value = 1.0
                }
            }

            return reward_value;
        },
        state_action_list_dict: state_action_list_dict_

    }
    setup_r_q_ = function(__self__, state_key, action_key)
    {
        if (state_action_list_dict_ != undefined && state_key in state_action_list_dict_)
        {
            state_action_list_dict_[state_key].push(action_key);
            state_action_list_dict_[state_key] = state_action_list_dict_[state_key].filter(function (x, i, self)
            {
                return self.indexOf(x) === i;
            });
        }
        else
        {
            if (state_action_list_dict_ == undefined) state_action_list_dict_ = {};
            state_action_list_dict_[state_key] = [];
            state_action_list_dict_[state_key].push(action_key);
        }
        q_value = __self__.extract_q_dict(state_key, action_key);
        __self__.save_q_dict(state_key, action_key, q_value);
        var r_value = __self__.extract_r_dict(state_key, action_key);
        r_value = r_value + 1.0;
        __self__.save_r_dict(state_key, r_value, action_key);
    }

    return constructor;

}) ();
