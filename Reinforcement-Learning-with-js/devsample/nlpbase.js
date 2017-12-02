/*
Repository name: accel-brain-code
Description: Base class of natural language processing.
Version: 1.0.1
Author: chimera0(RUM)
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0
Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)
*/


var NlpBase = (function()
{
    /**
     * @private
     *
     */
    var tinySegmenter_ = null;

    /**
     * @private
     *
     */
    var token_list_ = [];

    /**
     * Set Up hyperparams.
     *
     * @constructor
     */
    var constructor = function() {
        tinySegmenter_ = new TinySegmenter();
    }

    /** @constructor */
    constructor.prototype = {
        /**
         *
         * Tokenize document(string).
         *
         * @params {string}
         *
         */
        tokenize: function(__document__)
        {
            this.token_list_ = tinySegmenter_.segment(__document__);
        },
        token_list: token_list_
    }
}) ();
