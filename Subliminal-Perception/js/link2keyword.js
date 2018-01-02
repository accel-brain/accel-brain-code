/*
Library Name: link2keyword
Description: Your anchor link click event brings about the page transition and smooth scrolling to the position of highlight keyword.
Version: 1.0.1
Author: chimera0
Author URI: http://media.accel-brain.com/
License: GNU General Public License v2.0
*/

/*  Copyright 2016 chimera0 (email : kimera0kimaira@gmail.com)
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * Class of handling the click event.
 *
 */

var Link2Keyword = (function () {

    /** @private */
    var contentArea_ = "the-content";
    /** @private */
    var scrollSpeed_ = 1000; // millisecond
    /** @private */
    var localStorageName_ = "Link2Keyword";
    /** @private */
    var preLink2keyword_ = "";

    /**
     * Add anchor text as a hash to the anchor link.
     *
     * @private
     */
    setupHash_ = function () {
        url = window.location.href.split("#")[0];
        var targetA = $("#" + contentArea_ + " a");
        for (var i=0;i < targetA.length; i++) {
            if (targetA[i].href.indexOf("#") == -1) {
                targetA[i].href += "#link2keyword=" + encodeURIComponent(targetA[i].text);
            }
            targetURL = targetA[i].href.split("#")[0];
            if (targetURL == url)
            {
                targetA[i].className = "link-2-inner-keyword";
            }
        }
        targetA.bind("click", function () {
            var keyword = $(this).text();
            Link2Keyword.save (keyword);
        });
    }

    /**
     * Keyword Highlight.
     *
     * @private
     */
    highlight_ = function () {
        hash = location.hash;
        if (hash.indexOf("#link2keyword=") != -1) {
            keyword = hash.split("#link2keyword=")[1].split("&")[0];
            keyword = decodeURIComponent(keyword);
            try 
            {
                $("#" + contentArea_).removeHighlight();
            }
            catch (e)
            {
                console.log(e);
            }
            $("#" + contentArea_).highlight(keyword);
            preLink2keyword_ = keyword;
        }
    }

    /**
     * Smooth scrolling to highlight keyword.
     *
     * @private
     */
    scrolling_ = function () {
        var target = $("#" + contentArea_ + " .highlight");
        if (!target.offset()) {
            return false;
        }
        var windowHeight = $(window).height();
        var position = target.offset().top - (windowHeight / 2);
        $('body, html').animate({scrollTop:position}, scrollSpeed_, 'swing');
        return false;
    }

    save_ = function (keyword) {
        if(!localStorage) {
            console.log("localStorage error.");
            return false;
        }
        url = window.location.href.split("#")[0];
        var data = {
            "url": url,
            "preLink2keyword": preLink2keyword_,
            "link2keyword": keyword
        }
        var allData = JSON.parse(localStorage.getItem(localStorageName_));
        if (allData == undefined) allData = [];

        if (allData.length > 0) {
            allData.push(data);
        } else {
            allData = [data];
        }
        localStorage.setItem(localStorageName_, JSON.stringify(allData));
    }

    return {
        init: function (args) {
            if (args != undefined) {
                if ("contentArea" in args) contentArea_ = args["contentArea"];
                if ("scrollSpeed" in args) scrollSpeed_ = args["scrollSpeed"];
                if ("localStorageName" in args) localStorageName_ = args["localStorageName"];
            }
            setupHash_ ();
            $(function(){
                $(window).on('hashchange', function(){
                    Link2Keyword.highlight ();
                });
            });
        }, 
        highlight: highlight_,
        scrolling: scrolling_,
        save: save_
    }

}) ();
