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
var AutocompletionBoltzmannQLearning = (function()
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
    var constructor = function(strategy, nlp_base, n_gram, n) {
        strategy_ = strategy;
        nlp_base_ = nlp_base;
        n_gram_ = n_gram;
        if (n != undefined) n_ = n;
    }

    /** @constructor */
    constructor.prototype = {
        /**
         * Pre-training.
         *
         * @params{string}
         */
        pre_training: function (document)
        {
            this.nlp_base_.tokenize(document);
            token_list = this.nlp_base_.token;
            token_tuple_zip = this.n_gram_.generate_ngram_data_set(
                token_list,
                this.n_
            )
            for (token_tuple in token_tuple_zip)
            {
                this.setup_r_q_(token_tuple[0], token_tuple[1]);
            }
        }
    }
}) ();


