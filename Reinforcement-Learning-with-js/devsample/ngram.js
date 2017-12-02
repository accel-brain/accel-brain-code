/*
Repository name: accel-brain-code
Description: Base class of n-gram.
Version: 1.0.1
Author: chimera0(RUM)
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0
Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)
*/


var Ngram = (function()
{

    /**
     * Set Up hyperparams.
     *
     * @constructor
     */
    var constructor = function() {
    }

    /** @constructor */
    constructor.prototype = {
        /**
         *
         * Get n-gram.
         *
         * @params {array}
         * @params {int}
         *
         * @return {array}
         */
        generate_ngram_data_set: function(token_list, n)
        {
            if (n == undefined) n = 2;
            if (n >= token_list.length) return token_list;
            token_tuple_zip = [];
            for (var i =0;i<token_list.length;i++)
            {
                token_tuple = [];
                for(var j = 0;j<n;j++)
                {
                    token_tuple.push(token_list[i+j]);
                }
                token_tuple_zip.push(token_tuple);
            }
            return token_tuple_zip;
        }
    }
    return constructor;

}) ();
