/**
 * jQuery Mousehold.js 1.0
 * 
 * マウスの左クリックボタンの長押し動作にイベント（関数）を割り当てるプラグインです。
 * 
 * 《使い方》
 * .mousehold(func, options) 戻り値: jQueryオブジェクト
 * 
 * //長押し判定させる要素を取得
 * var button1 = $('#button-1'); //ボタン要素でなくてもDOM要素なら何でもOK
 * 
 * // 第1引数にコールバック関数を、第2引数にオプション（省略可）を指定。コールバックが呼び出されると、jQueryイベントと同様にイベントオブジェクトが第1引数に渡される
 * button1.mousehold(func, options);
 * 
 * //このような書き方も可能。上記と同じ意味
 * button1.mousehold.add(func, options);
 * 
 * //イベントハンドラを解除するとき
 * button1.mousehold.remove();
 * 
 * オプションはハッシュ{}で指定
 * 
 * bindObject: コールバック関数のthisキーワードに割り当てるオブジェクトを指定。デフォルトはnull（=Windowオブジェクト）。特別に文字列で"self"と指定するとイベント元の要素がバインドされる。
 * timeout: マウスボタン長押し後、イベントが発生するまでのホールド期間を指定する。ミリ秒単位なので1秒なら1000を指定する（デフォルトは1000）。
 * 
 */

;(function($) {

    var defaultOptions = {
        bindObject: null,
        timeout: 1000,
    };

    Object.defineProperty($.fn, 'mousehold', {
        get: function () {
            var initializer = this.data('mousehold-initializer');

            if (!initializer) {
                initializer = $.proxy(init, this);

                initializer.add = $.proxy(add, this);
                initializer.remove = $.proxy(remove, this);

                this.data('mousehold-initializer', initializer);
                this.data('mousehold-initialized', false);
            }

            return initializer;
        },
    });

    function init(callback, options) {
        var data = {
            callback: noop,
            options: $.extend({}, defaultOptions),
            timerId: 0,
            holding: false,
        };

        this.data('mousehold-initialized', true);
        this.data('mousehold', data);

        this.data('mousehold-initializer').add(callback, options);

        return this;
    }

    function add(callback, options) {
        var self, data;

        if (!this.data('mousehold-initialized')) {
            return this.data('mousehold-initializer')(callback, options);
        }

        self = this;
        data = this.data('mousehold');

        if ($.isFunction(callback)) {
            data.callback = callback;

            if ($.isPlainObject(options)) {
                data.options = $.extend(data.options, options);
            }
        } else if ($.isPlainObject(callback)) {
            data.options = $.extend(data.options, callback);
        }

        if (data.options.bindObject === 'self') {
            data.options.bindObject = this;
        }

        this.data('mousehold', data);

        return this.each(function () {
            self.on('mousedown.mousehold touchstart.mousehold', $.proxy(setTimer, self)).
                on('mouseup.mousehold touchend.mousehold', $.proxy(clearTimer, self)).
                on('mouseout.mousehold touchcancel.mousehold', $.proxy(clearTimer, self));
        });
    }

    function remove() {
        var self = this;
        this.removeData(['mousehold', 'mousehold-initializer', 'mousehold-initialized']);

        return this.each(function () {
            self.off('mousedown.mousehold touchstart.mousehold').
                off('mouseup.mousehold touchend.mousehold').
                off('mouseout.mousehold touchcancel.mousehold');
        });
    }

    function setTimer(e) {
        var data = this.data('mousehold');
        if (!data.holding) {
            data.holding = true;
            data.timerId = window.setTimeout(function () {
                if (data.holding) {
                    data.holding = false;
                    window.clearTimeout(data.timerId);
                    data.callback.call(data.options.bindObject, e);
                }
            }, data.options.timeout);
        }
    }

    function clearTimer() {
        var data = this.data('mousehold');
        if (data.holding) {
            data.holding = false;
            window.clearTimeout(data.timerId);
        }
    }

    function noop() {}

})(jQuery);