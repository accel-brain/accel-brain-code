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
    var token_ = [];

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
            __document__ = __document__.replace(/\r?\n/g, "");
            if ((__document__.match(/^[^\x01-\x7E\xA1-\xDF]+$/) != null) == true)
            {
                var token_list = tinySegmenter_.segment(__document__);
            }
            else
            {
                var token_list = __document__.split(" ");
                var token = [];
                for (var i = 0;i<token_list.length;i++)
                {
                    if (token_list[i] != " " && token_list[i] != undefined)
                    {
                        token.push(" " + token_list[i])
                    }
                }
                token_list = token;
            }
            this.token = token_list;
        },
        token: token_
    }
    return constructor;

}) ();
