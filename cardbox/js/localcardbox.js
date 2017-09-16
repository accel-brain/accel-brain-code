/*
Repository name: accel-brain-code
Description: This is a demo code for my case study in the context of my website.
Version: 1.0.1
Author: chimera0
Author URI: http://accel-brain.com/
License: GNU General Public License v2.0

Copyright 2017 chimera0 (email : ai-brain-lab@accel-brain.com)

*/

/**
 * Base Class of the cardbox.
 *
 */
var LocalCardBox = (function () {

    /** @private */
    var cardBoxDict_ = new Object();
    /** @private */
    var maxZ_ = 1000;
    /** @private */
    var cardClassName_ = "card";
    /** @private */
    var idPrefix_ = "card";

    /**
     * Create card.
     *
     * @private
     * @param {string} text.
     * @param {number} X axis.
     * @param {number} Y axis.
     */
    create_ = function (text, x, y)
    {
        var date = new Date() ;
        var timestamp = Math.floor(date.getTime() / 1000);
        var cardId = timestamp;
        cardId = idPrefix_ + cardId;
        make_ (cardId, text, x, y);
    }

    /**
     * Create card.
     *
     * @private
     * @param {string} uniquie id.
     * @param {string} text.
     * @param {number} X axis.
     * @param {number} Y axis.
     */
    make_ = function (cardId, text, x, y)
    {
        console.log("make");
        var cardObj = $('<div>');
        cardObj.addClass("ui-widget ui-widget-content " + cardClassName_);
        cardObj.attr('id', cardId);
        cardObj.text(text);
        cardObj.css({top:y, left:x});
        cardObj.appendTo("body");

        cardObj.draggable({
            start: function() {
                maxZ_ ++;
                cardBoxDict_[this.id]["z"] = maxZ_;
                $("#" + this.id).css("zIndex", maxZ_);
            },
            drag: function() {
                cardBoxDict_[cardId]["drag"] = true;
            },
            stop: function() {
                var x = document.getElementById(this.id).offsetLeft;
                var y = document.getElementById(this.id).offsetTop;
                cardBoxDict_[this.id]["x"] = x;
                cardBoxDict_[this.id]["y"] = y;
                cardBoxDict_[cardId]["drag"] = false;
            }
        });

        cardObj.dblclick(function()
        {
            if (cardBoxDict_[cardId]["edit"] == false)
            {
                var _cardText = cardObj.text();
                var cardTextArea = $('<textarea>');
                cardTextArea.attr('id', cardId + "--text-area");
                cardTextArea.val(_cardText);
                cardTextArea.addClass("card-text-area");
                cardObj.html(cardTextArea);
                cardBoxDict_[cardId]["edit"] = true;

                $('#' + cardId + "--text-area").blur(function ()
                {
                    var cardId = this.id.split("--")[0];
                    update_(cardId);
                });
            }
        });

        cardObj.mousehold(function()
        {
            if (cardBoxDict_[cardId]["drag"] == true)
            {
                return;
            }

            var dialogObj = $("<div>");
            dialogObj.text("This operation cannot be undone.");
            dialogObj.attr('id', "cardDeleteDialog");
            dialogObj.appendTo("body");

            $('#cardDeleteDialog').dialog({
                autoOpen: true,
                title: 'Delete this card ?',
                closeOnEscape: false,
                modal: true,
                buttons: {
                    "Yes": function(){
                        delete_(cardId);
                        $('#cardDeleteDialog').remove();
                    },
                    "No": function(){
                        $(this).dialog('close');
                        $('#cardDeleteDialog').remove();
                    }
                }
            });
        });

        cardBoxDict_[cardId] = {
            'text': text,
            'x': x,
            'y': y,
            'z': maxZ_,
            'edit': false,
            'drag': false
        };
    }

    /**
     * Update card text.
     *
     * @private
     * @param {string} unique id.
     */
    update_ = function (cardId)
    {
        if (cardBoxDict_[cardId]["edit"] == true)
        {
            var cardObj = $('#' + cardId);
            var text = $('#' + cardId + "--text-area").val();
            cardObj.text(text);
            cardBoxDict_[cardId]["text"] = text;
            cardBoxDict_[cardId]["edit"] = false;
        }
    }

    /**
     * Delete card.
     *
     * @private
     * @param {string} unique id.
     */
    delete_ = function (cardId)
    {
        if (cardBoxDict_[cardId])
        {
            $("#" + cardId).remove();
        }
    }

    /**
     * Refresh all card data.
     *
     * @private
     */
    refresh_ = function ()
    {
        $("." + cardClassName_).remove();
        cardBoxDict_ = new Object();
    }

    /**
     * Export object of cardBoxDict_
     *
     * @private
     */
    export_ = function ()
    {
        return cardBoxDict_;
    }

    /**
     * Set Up.
     *
     * @constructor
     * @params {object}
     */
    var constructor = function(params)
    {
        var createInputId = false;
        var createButtonId = false;
        var saveButtonId = false;
        if (params != undefined)
        {
            if (params["cardClassName"])
            {
                cardClassName_ = params["cardClassName_"];
            }
            if (params["createInputId"])
            {
                createInputId = params["createInputId"];
            }
            if (params["createButtonId"])
            {
                createButtonId = params["createButtonId"];
            }
        }
        $(function() {
            if (createInputId != false && createButtonId != false)
            {
                $("#" + createButtonId).click(function (e) {
                    var text = $("#" + createInputId).val();
                    var x = e.pageX;
                    var y = e.pageY;
                    create_ (text, x, y, false);
                    $("#" + createInputId).val("");
                });
            }

            refresh_ ();
        });
    }

    /** @constructor */
    constructor.prototype = {
        create: create_,
        refresh: refresh_,
        export: export_
    };

    return constructor;

}) ();
