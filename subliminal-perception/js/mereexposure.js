/*
Repository name: accel-brain-code
Description: This is a demo code for my case study in the context of my website.
Version: 1.0.1
Author: chimera0
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0

Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)

*/

var MereExposure = (function()
{
    /** 
     * @private
     *
     * Stimulus presentation interval(milli seconds).
     */
    intervalMiliSec_ = 100;

    /** 
     * @private
     *
     * Words to be presented.
     */
    keywordList_ = ['Demo', 'of', 'Mere', ' Exposure', 'Effect'];

    /** 
     * @private
     *
     * Background color code during presentation.
     */
    backgroundColorCode_ = "#000";

    /**
     * @private
     *
     * Make background color transparent.
     */
    backgroundOpacity_ = 1.0;

    /**
     * @private
     *
     * Background Image.
     */
    backgroundImage_ = null;

    /** 
     * @private
     *
     * Color code of words to be presented.
     */
    colorCode_ = "#555";

    /** 
     * @private
     *
     * Minimum value of words to be presented.
     */
    minFontSize_ = 18;

    /** 
     * @private
     *
     * Maximum value of words to be presented.
     */
    maxFontSize_ = 45;

    /** 
     * @private
     *
     * Delete all other ot not.
     */
    deleteAllOther_ = false;

    /** 
     * @private
     *
     * Reset background sheet or not.
     */
    resetBgSheet_ = false;

    /** 
     * @private
     *
     * min top.
     */
    minTop_ = 0;

    /** 
     * @private
     *
     * Decide the x postion of each word to be presented.
     */
    decideRandomX_ = function()
    {
        width = $(window).width();
        return Math.round(Math.random() * width);
    }

    /** 
     * @private
     *
     * Decide the y postion of each word to be presented.
     */
    decideRandomY_ = function ()
    {
        height = $(window).height();
        return (Math.round(Math.random() * height)) + minTop_;
    }

    /** 
     * @private
     *
     * Decide font size of each word to be presented.
     */
    decideRandomFontSize_ = function ()
    {
        return Math.round(minFontSize_ + Math.random() * maxFontSize_);
    }

    /** 
     * @private
     *
     * Callback method.
     */
    callback_ = function () {}

    /**
     * Set Up.
     *
     * @constructor
     * @params {object}
     */
    var constructor = function(params) {
        if (params != undefined)
        {
            if ("intervalMiliSec" in params)
            {
                intervalMiliSec_ = params.intervalMiliSec;
            }
            if ("keywordList" in params)
            {
                keywordList_ = params.keywordList;
            }
            if ("backgroundColorCode" in params)
            {
                backgroundColorCode_ = params.backgroundColorCode;
            }
            if ("colorCode" in params)
            {
                colorCode_ = params.colorCode;
            }
            if ("minFontSize" in params)
            {
                minFontSize_ = params.minFontSize;
            }
            if ("maxFontSize" in params)
            {
                maxFontSize_ = params.maxFontSize;
            }
            if ("callback" in params)
            {
                callback_ = params.callback;
            }
            if ("deleteAllOther" in params)
            {
                deleteAllOther_ = params.deleteAllOther;
            }
            if ("resetBgSheet" in params)
            {
                resetBgSheet_ = params.resetBgSheet;
            }
            if ("backgroundOpacity" in params)
            {
                backgroundOpacity_ = params.backgroundOpacity;
            }
            if ("backgroundImage" in params)
            {
                backgroundImage_ = params.backgroundImage;
            }
        }
    }

    /** @constructor */
    constructor.prototype = {

        /**
         * Execute Mere Exposure Effect.
         *
         */
        effect: function (keywordList)
        {
            if (keywordList != undefined)
            {
                keywordList_ = keywordList;
            }

            if (deleteAllOther_)
            {
                $("html").empty();
                $("html").append($("<body></body>"));
                $("body").css({"background-color": backgroundColorCode_});
            }
            var scrollTop = $(window).scrollTop();
            var divHeight = $(window).height();
            var divTop = scrollTop;
            minTop_ = divTop;
            var div = $("<div></div>", {
                css: {
                    "width": $(window).width(),
                    "height": divHeight,
                    "z-index": "10000px",
                    "background-color": backgroundColorCode_,
                    "position": "absolute",
                    "left": 0,
                    "top": divTop,
                    "opacity": backgroundOpacity_
                }
            });
            div.attr("id", "MereExposureEffectSheet");
            if (backgroundImage_ != null)
            {
                div.css ({
                    "background-repeat": "no-repeat",
                    "background-image": 'url(' + backgroundImage_ + ')',
                    "background-position": "center center"
                });
            }
            $("body").append(div)
            timer = setInterval(function()
            {
                if (keywordList_.length > 0)
                {
                    if (document.getElementById("MereExposureEffect"))
                    {
                        $("#MereExposureEffect").remove();
                    }
                    var x = decideRandomX_();
                    var y = decideRandomY_();
                    console.log(x)
                    console.log(y)
                    var fontSize = decideRandomFontSize_();
                    var span = $("<span></span>", {
                        id: "MereExposureEffect",
                        css: {
                            "position": "absolute",
                            "top": y + "px",
                            "left": x + "px",
                            "z-index": "1000001px",
                            "color": colorCode_,
                            "font-size": fontSize + "px"
                        }
                    });
                    span.html(keywordList_.shift());
                    $("body").append(span);
                }
                else
                {
                    clearInterval(timer);
                    if (document.getElementById("MereExposureEffect"))
                    {
                        $("#MereExposureEffect").remove();
                    }
                    if (resetBgSheet_)
                    {
                        if (document.getElementById("MereExposureEffectSheet"))
                        {
                            $("#MereExposureEffectSheet").remove();
                        }
                    }
                    callback_();
                }
            }, intervalMiliSec_);
        }
    };

    return constructor;
})();
