/*
Repository name: accel-brain-code
Description: Controller of auto completion.
Version: 1.0.1
Author: chimera0(RUM)
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0
Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)
*/

var Controller = (function() {
    /*
     * @private
     *
     */
    autocompletion_ = null;

    /*
     * @private
     *
     */
    boltzmann_ = null;

    /*
     * @private
     *
     */
    q_learning_ = null;

    /*
     * @private
     *
     */
    input_memroy_ = "";

    /*
     * @pricate
     *
     */
    local_storage_dict_ = {
        "r_dict": "autocompletion__r_dict_20171229",
        "q_dict": "autocompletion__q_dict_20171229"
    }

    var latest_state_key_ = null;
    var limit_ = 5;

    /**
     * Set Up hyperparams.
     *
     * @params{object}
     *
     * @constructor
     */
    var constructor = function(params) {
        var nlp_base = new NlpBase({
            "jpOnlyFlag": params.jpOnlyFlag
        });
        var n_gram = new Ngram();
        var autocompletion = new Autocompletion(
            nlp_base,
            n_gram,
            params.n
        );
        var boltzmann = new Boltzmann(
            autocompletion,
            {
                "time_rate": params.time_rate
            }
        );

        var q_learning = new QLearning(
            boltzmann,
            {
                "alpha_value": params.alpha_value,
                "gamma_value": params.gamma_value
            }
        );
        if (this.check_memory() == true)
        {
            this.recall();
        }
        if (params.document != null)
        {
            autocompletion.pre_training(q_learning, params.document);
        }

        limit_ = params.limit;

        this.autocompletion_ = autocompletion;
        this.boltzmann_ = boltzmann;
        this.q_learning_ = q_learning;

        $(window).on("beforeunload", function() {
            try
            {
                localStorage.setItem(
                    local_storage_dict_["r_dict"],
                    JSON.stringify(this.q_learning_.r_dict)
                );
                localStorage.setItem(
                    local_storage_dict_["q_dict"],
                    JSON.stringify(this.q_learning_.q_dict)
                );
            }
            catch(e)
            {
                console.log(e);
            }
        }.bind(this));
    }

    /** @constructor */
    constructor.prototype = {
        q_learning: q_learning_,
        latest_state_key: latest_state_key_,
        limit: limit_,

        recommend_list : function(input_document)
        {
            var state_key = this.autocompletion_.lap_extract_ngram(
                this.q_learning_,
                input_document
            );

            latest_state_key_ = state_key;
            this.limit = limit_;
            setTimeout(function() {
                this.q_learning_.learn(this.latest_state_key_, this.limit);
            }.bind(this), 5000);

            var next_action_list = this.q_learning_.extract_possible_actions(
                state_key
            );

            var q_a_list = [];
            for (var i = 0;i<next_action_list.length;i++)
            {
                var q_value = this.q_learning_.extract_q_dict(state_key, next_action_list[i]);
                q_a_list.push([q_value, next_action_list[i]]);
            }
            q_a_list = q_a_list.sort(function(a, b) {
                if (a[0] !== b[0])
                {
                    return a[0] - b[0];
                }
                else
                {
                    return a[1] - b[1];
                }
            });
            var recommended_action_list = [];
            for (var i = 0;i<q_a_list.length;i++)
            {
                recommended_action_list.push(q_a_list[i][1]);
            }
            return recommended_action_list;
        },
        /**
         *
         * Tokenize document(string).
         *
         * @params {string}
         *
         * @return {string}
         *
         */
        recommend : function(input_document)
        {
            var state_key = this.autocompletion_.lap_extract_ngram(
                this.q_learning_,
                input_document
            );
            this.q_learning_.learn(state_key, this.limit_);
            var next_action_list = this.q_learning_.extract_possible_actions(
                state_key
            );
            var action_key = this.q_learning_.select_action(
                state_key,
                next_action_list
            );
            var reward_value = this.q_learning_.observe_reward_value(
                state_key,
                action_key
            );
            var q_value = this.q_learning_.extract_q_dict(
                state_key,
                action_key
            );
            input_memroy_ = input_memroy_ + input_document;
            this.autocompletion_.pre_training(
                this.q_learning_,
                input_memroy_
            );
            return action_key;
        },
        check_memory: function()
        {
            try
            {
                var storage_keys_list = Object.keys(local_storage_dict_);
                var flag = true;
                for (var i = 0;i<storage_keys_list.length;i++)
                {
                    if (localStorage.getItem(local_storage_dict_[storage_keys_list[i]]) == null)
                    {
                        flag = false;
                        break;
                    }
                }
                return flag;
            }
            catch (e)
            {
                console.log(e);
                return false;
            }
        },
        memorize: function ()
        {
            try
            {
                localStorage.setItem(
                    local_storage_dict_["r_dict"],
                    JSON.stringify(this.q_learning_.r_dict)
                );
                var q_dict = this.q_learning_.dump_q_dict()
                localStorage.setItem(
                    local_storage_dict_["q_dict"],
                    JSON.stringify(this.q_learning_.q_dict)
                );
            }
            catch(e)
            {
                console.log(e);
            }
        },
        recall: function ()
        {
            try
            {
                var r_dict = localStorage.getItem(
                    local_storage_dict_["r_dict"]
                );
                var q_dict = localStorage.getItem(
                    local_storage_dict_["q_dict"]
                );
                if (r_dict != null)
                {
                    this.q_learning_.r_dict = JSON.parse(r_dict);
                }
                if (q_dict != null)
                {
                    this.q_learning_.q_dict = JSON.parse(q_dict);
                }
            }
            catch(e)
            {
                console.log(e);
            }
        },
        add_learn: function (__document__)
        {
            this.autocompletion_.pre_training(
                this.q_learning_,
                __document__
            );
        }
    }

    return constructor;

}) ();
