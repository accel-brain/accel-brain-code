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

    /**
     * Set Up hyperparams.
     *
     * @params{object}
     *
     * @constructor
     */
    var constructor = function(params) {
        var nlp_base = NlpBase();
        var n_gram = Ngram();
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
        autocompletion.pre_training(q_learning, params.document);

        limit_ = params.limit;

        autocompletion_ = autocompletion;
        boltzmann_ = boltzmann;
        q_learning_ = q_learning;
    }

    /** @constructor */
    constructor.prototype = {
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
            next_action_list = this.q_learning_.extract_possible_actions(
                state_key
            );
            action_key = this.q_learning_.select_action(
                state_key,
                next_action_list
            );
            reward_value = this.q_learning_.observe_reward_value(
                state_key,
                action_key
            );
            q_value = q_learning.extract_q_dict(
                state_key,
                action_key
            );
            this.autocompletion_.pre_training(
                this.q_learning_,
                input_document
            );
            return action_key;
        }
    }
    return constructor;

}) ();


