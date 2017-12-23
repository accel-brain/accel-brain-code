/*
Library Name: Binaural Beat and Monaural Beat with Web Audio API
Description: This application enables you to handle your mind state by a kind of "Brain-Wave Controller". You can post blog while controlling your brain waves.
Version: 1.0.1
Author: chimera0
Author URI: https://accel-brain.com/
License: GPL2
*/

/*  Copyright 2017 chimera0(RUM) (email : ai-brain-lab@accel-brain.com)
 
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License, version 2, as
     published by the Free Software Foundation.
 
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
 
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

/**
 * Class of handling the Binaural Beat or Monaural Beat.
 *
 * @param {number} 0: for left sound 1: for right sound. 2: for both sound. 
 *
 */
var BuildBeat = (function() {

    /** @private */
    var contextDict_ = {};
    /** @private */
    var oscillatorDict_ = {};
    /** @private */
    var processorDict_ = {};
    /** @private */
    var isPlayingFlags_ = {};

    /**
     * Build instances in connection with Web Audio API.
     *
     * @private
     * @param {number} 0: Left sound 1: Right sound 2: Both
     */
    function init_ (LR) {
        if (LR == 0 || LR == 1 || LR == 2) {
            createContext_(LR);
            createOscillator_(LR);
        } else {
            console.log("The value of LR must be 0 or 1 or 2.")
        }
    };

    /**
     * Instantiate AudioContext.
     *
     * @private
     * @param {number} 0: Left sound 1: Right sound 2: Both
     */
    function createContext_ (LR) {
        window.AudioContext = window.AudioContext || window.webkitAudioContext;
        try {
            if (!contextDict_[LR]) {
                var context = new AudioContext();
                context.createGain = context.createGain || context.createGainNode;
                contextDict_[LR] = context;
            }
        } catch (error) {
            alert(error.message + ' : Web browser (such as Google Chorome, FireFox or Safari) is required.');
            return;
        }
    };

    /**
     * Instantiate Oscillator of AudioContext.
     *
     * @private
     * @param {number} 0: Left sound 1: Right sound 2: Both
     */
    function createOscillator_ (LR) {
        var oscillator = contextDict_[LR].createOscillator();
        oscillatorDict_[LR] = oscillator;
    }

    /**
     * Select buffer size. It depends on each browser.
     *
     * @private
     * @return {number} buffer size.
     */
    function getBufferSize_ () {
        // Windows 8
        if (/(Win(dows )?NT 6\.2)/.test(navigator.userAgent)) {
            return 1024;
        // Windows 7
        } else if (/(Win(dows )?NT 6\.1)/.test(navigator.userAgent)) {
            return 1024;
        // Windows Vista
        } else if (/(Win(dows )?NT 6\.0)/.test(navigator.userAgent)) {
            return 2048;
        // Windows XP
        } else if (/Win(dows )?(NT 5\.1|XP)/.test(navigator.userAgent)) {
            return 4096;
        // Mac OS X
        } else if (/Mac|PPC/.test(navigator.userAgent)) {
            return 1024;
        // Linux
        } else if (/Linux/.test(navigator.userAgent)) {
            return 8192;
        // iOS
        } else if (/iPhone|iPad|iPod/.test(navigator.userAgent)) {
            return 2048;
        // Otherwise
        } else {
            return 16384;
        }
    };

    /**
     * Control volumes.
     *
     * @private
     * @param {number} 0: left volume 1: right volume 2: both
     * @param {number} value of volume.
     */
    function contorolVolumeLR_(LR, volume) {
        if (LR == 0 || LR == 1 || LR == 2) {
            /* 0: left 1: right 2: both */

            var splitter = contextDict_[LR].createChannelSplitter(2);
            var merger = contextDict_[LR].createChannelMerger(2);
            var gainL = contextDict_[LR].createGain();
            var gainR = contextDict_[LR].createGain();

            var min = gainL.gain.minValue || 0;
            var max = gainL.gain.maxValue || 1;

            if ((volume < min) || (volume > max)) {
                return;
            }

            if (LR == 0) {
                gainL.gain.value = volume;
                gainR.gain.value = 0.0;
            } else if (LR == 1) {
                gainL.gain.value = 0.0;
                gainR.gain.value = volume;
            } else {
                gainL.gain.value = volume;
                gainR.gain.value = volume;
            }

            if (processorDict_[LR] == undefined) {
                var processor = contextDict_[LR].createScriptProcessor(getBufferSize_(), 2, 2);

                processor.onaudioprocess = function(event) {
                    onAudioProcess_(event);
                };
                processorDict_[LR] = processor
                oscillatorDict_[LR].connect(processorDict_[LR]);
                processorDict_[LR].connect(splitter);
                splitter.connect(gainL, 0, 0);
                splitter.connect(gainR, 1, 0);
                gainL.connect(merger, 0, 0);
                gainR.connect(merger, 0, 1);
                merger.connect(contextDict_[LR].destination);
            }
        }
    }

    /**
     * Process input and output buffer.
     *
     * @private
     * @param {AudioProcessingEvent} An event, implementing the AudioProcessingEvent interface.
     */
    function onAudioProcess_ (event) {
        var inputLs  = event.inputBuffer.getChannelData(0);
        var inputRs  = event.inputBuffer.getChannelData(1);
        var outputLs = event.outputBuffer.getChannelData(0);
        var outputRs = event.outputBuffer.getChannelData(1);
        outputLs.set(inputLs);
        outputRs.set(inputRs);
    };

    /**
     * Select sound source.
     *
     * @constructor
     * @param {number} 0: left 1:right 2:both
     */
    var constructor = function(LR) {
        if (LR == 0 || LR == 1 || LR == 2) {
            /* 0: left 1: right 2: both */
           this.LR = LR;
           init_(LR);
        } else {
            console.log("The value of LR must be 0 or 1 or 2.");
            return;
        }
    };

    /** @constructor */
    constructor.prototype = {

        /**
         * Control the sound volume.
         *
         * @param {number} the sound volume.
         */
        controlVolume: function (volume) {
            if (isPlayingFlags_[this.LR] != true) {
                volume = Number(volume);
                contorolVolumeLR_(this.LR, volume);
            }
        },

        /**
         * Change the wave type.
         *
         * @param {string} sine, square, sawtooth, or triangle.
         */
        selectWaveType: function (type) {
            oscillatorDict_[this.LR].type = type;
        }, 

        /**
         * Tuning the frequency. (0 - 1000 Hz)
         *
         * @param {number} frequency.
         */
        controlFrequency: function (frequency) {
            var min = oscillatorDict_[this.LR].frequency.minValue || 0;
            var max = oscillatorDict_[this.LR].frequency.maxValue || 1000;
 
            frequency = Number(frequency);
            if ((frequency >= min) && (frequency <= max)) {
                oscillatorDict_[this.LR].frequency.value = frequency;
            }
        },

        /**
         * Tuning an a-rate AudioParam.
         *
         * @param {number} value of the detune.
         */
        controlDetune: function (detune) {
            var min = oscillatorDict_[this.LR].detune.minValue || -4800;
            var max = oscillatorDict_[this.LR].detune.maxValue ||  4800;

            detune = Number(detune);
            if ((detune >= min) && (detune <= max)) {
                oscillatorDict_[this.LR].detune.value = detune;
            }
        },

        /**
         * Start playing Binaural or Monoaural beat.
         *
         */
        playBeat: function () {
            if (isPlayingFlags_[this.LR] == true) {
                return;
            }
            oscillatorDict_[this.LR].start = oscillatorDict_[this.LR].start || oscillatorDict_[this.LR].noteOn;
            oscillatorDict_[this.LR].stop  = oscillatorDict_[this.LR].stop  || oscillatorDict_[this.LR].noteOff;

            if (processorDict_[this.LR] == undefined) {
                contorolVolumeLR_(this.LR, 1.0);
            }
            oscillatorDict_[this.LR].start(0);
            isPlayingFlags_[this.LR] = true;
            ComposeForm.disabledVolume(true);
        },

        /**
         * Stop playing the beat.
         *
         */
        stopBeat: function () {
            if (isPlayingFlags_[this.LR] == false) {
                return;
            }
            oscillatorDict_[this.LR].stop(0); 
            isPlayingFlags_[this.LR] = false;
            processorDict_[this.LR] = undefined;
            init_(this.LR);
            ComposeForm.disabledVolume(false);
        },

        /**
         * Check the state of playing.
         *
         * @return {boolean} true: playing false: not playing.
         */
        isPlaying: function () {
            return isPlayingFlags_[this.LR];
        }
    };

    return constructor;

}) ();

/**
 * Make web form and fieldset for playing Binaural or Monoral Beat.
 *
 */
var ComposeForm = (function () {

    /** @private */
    var sineOnlyFlag_ = true;

    /** @private */
    var startOrStopButtonId_ = "start-or-stop";
    /** @private */
    var volumeRangeId_ = "range-volume";
    /** @private */
    var volumeDisplayId_ = "output-volume";
    /** @private */
    var waveFormFormId_ = "form-wave-type";
    /** @private */
    var waveTypeRadioId_ = "radio-wave-type";
    /** @private */
    var leftFrequencyRangeId_ = "left-range-frequency";
    /** @private */
    var rightFrequencyRangeId_ = "right-range-frequency";
    /** @private */
    var leftFrequencyDisplayId_ = "left-output-frequency";
    /** @private */
    var rightFrequencyDisplayId_ = "right-output-frequency";
    /** @private */
    var detuneRangeId_ = "range-detune";
    /** @private */
    var detuneDisplayId_ = "output-detune";
    /** @private */
    var beatTypeFormId_ = "form-beat-type";
    /** @private */
    var beatTypeRangeId_ = "radio-beat-type";

    return {
        /**
         * The dom ID of start or stop button.
         * @type {string}
         */
        startOrStopButtonId: startOrStopButtonId_,

        /**
         * The dom ID of the range for tuning volume.
         * @type {string}
         */
        volumeRangeId: volumeRangeId_,

        /**
         * The dom ID of displaying now volume.
         * @type {string}
         */
        volumeDisplayId: volumeDisplayId_,

        /**
         * The dom ID of form for selecting the wave type.
         * @type {string}
         */
        waveFormFormId: waveFormFormId_,

        /**
         * The dom ID of radio box for selecting the wave type.
         * @type {string}
         */
        waveTypeRadioId: waveTypeRadioId_,

        /**
         * The dom ID of the range for tuning left frequency.
         * @type {string}
         */
        leftFrequencyRangeId: leftFrequencyRangeId_,

        /**
         * The dom ID of the range for tuning right frequency.
         * @type {string}
         */
        rightFrequencyRangeId: rightFrequencyRangeId_,

        /**
         * The dom ID of the span for displaying now left frequency.
         * @type {string}
         */
        leftFrequencyDisplayId: leftFrequencyDisplayId_,

        /**
         * The dom ID of the span for displaying now right frequency.
         * @type {string}
         */
        rightFrequencyDisplayId: rightFrequencyDisplayId_,

        /**
         * The dom ID of the range for tuning detune.
         * @type {string}
         */
        detuneRangeId: detuneRangeId_,

        /**
         * The dom ID of the span for displaying now detune.
         * @type {string}
         */
        detuneDisplayId: detuneDisplayId_,

        /**
         * The dom ID of form for selecting the wave type.
         * @type {string}
         */
        beatTypeFormId: beatTypeFormId_,

        /**
         * The dom ID for displaying now wave type.
         * @type {string}
         */
        beatTypeRangeId: beatTypeRangeId_,

        /**
         * Restrict tuning volume.
         *
         * @param {boolean} true: disabled. false: not disabled.
         *
         * TODO(chimera0): make it possible to tune the volumes while playing.
         */
        disabledVolume: function(_bool) {
            if (_bool) {
                document.getElementById(volumeRangeId_).setAttribute("disabled", "disabled");
            } else {
                document.getElementById(volumeRangeId_).removeAttribute("disabled");
            }
        },

        /**
         * Make web form and fieldset.
         *
         * This indent style is unified with not Google JavaScript Style Guide 
         *      but rather Python.
         *
         * @return {Element} web form and fieldset.
         */
        createFieldset: function () {
            fieldset = document.createElement("fieldset");
                dl = document.createElement("dl");
                    dt = document.createElement("dt");
                    label = document.createElement("label");
                    span = document.createElement("span");
                    span.innerHTML = "Beat Type: ";
                    label.appendChild(span);
                    dt.appendChild(label);
                dl.appendChild(dt);
                    dd = document.createElement("dd");
                        form = document.createElement("form");
                        form.id = beatTypeFormId_;
                        form.setAttribute("name", beatTypeFormId_);
                            p = document.createElement("p");
                                label = document.createElement("label");
                                    input = document.createElement("input");
                                    input.setAttribute("type", "radio");
                                    input.setAttribute("name", beatTypeRangeId_);
                                    input.setAttribute("value", "0");
                                    input.setAttribute("checked", "checked");
                                label.appendChild(input);
                                    span = document.createElement("span");
                                    span.innerHTML = "Binaural Beat";
                                label.appendChild(span);
                            p.appendChild(label);
                        form.appendChild(p);
                            p = document.createElement("p");
                                label = document.createElement("label");
                                    input = document.createElement("input");
                                    input.setAttribute("type", "radio");
                                    input.setAttribute("name", beatTypeRangeId_);
                                    input.setAttribute("value", "1");
                                label.appendChild(input);
                                    span = document.createElement("span");
                                    span.innerHTML = "Monoral Beat";
                                label.appendChild(span);
                            p.appendChild(label)
                        form.appendChild(p);
                    dd.appendChild(form);
                dl.appendChild(dd)
            if (!sineOnlyFlag_) {
                    dt = document.createElement("dt");
                    dt.innerHTML = "Wave Form: ";
                dl.appendChild(dt);
                    dd = document.createElement("dd");
                        form = document.createElement("form");
                        form.id = waveFormFormId_;
                        form.setAttribute("name", waveFormFormId_);
                            p = document.createElement("p");
                                label = document.createElement("label");
                                    input = document.createElement("input");
                                    input.setAttribute("type", "radio");
                                    input.setAttribute("name", waveTypeRadioId_);
                                    input.setAttribute("value", "sine");
                                    input.setAttribute("checked", "checked");
                                label.appendChild(input);
                                    span = document.createElement("span");
                                    span.innerHTML = "Sine Wave";
                                label.appendChild(span);
                            p.appendChild(label);
                        form.appendChild(p);
                            p = document.createElement("p");
                                label = document.createElement("label");
                                    input = document.createElement("input");
                                    input.setAttribute("type", "radio");
                                    input.setAttribute("name", waveTypeRadioId_);
                                    input.setAttribute("value", "square");
                                label.appendChild(input);
                                    span = document.createElement("span");
                                    span.innerHTML = "Square Wave";
                                label.appendChild(span);
                            p.appendChild(label);
                        form.appendChild(p);
                            p = document.createElement("p");
                                label = document.createElement("label");
                                    input = document.createElement("input");
                                    input.setAttribute("type", "radio");
                                    input.setAttribute("name", waveTypeRadioId_);
                                    input.setAttribute("value", "sawtooth");
                                label.appendChild(input);
                                    span = document.createElement("span");
                                    span.innerHTML = "Sawtooth Wave";
                                label.appendChild(span);
                            p.appendChild(label);
                        form.appendChild(p);
                            p = document.createElement("p");
                                label = document.createElement("label");
                                    input = document.createElement("input");
                                    input.setAttribute("type", "radio");
                                    input.setAttribute("name", waveTypeRadioId_);
                                    input.setAttribute("value", "triangle");
                                label.appendChild(input);
                                    span = document.createElement("span");
                                    span.innerHTML = "Triangle Wave";
                                label.appendChild(span);
                            p.appendChild(label);
                        form.appendChild(p);
                    dd.appendChild(form);
                dl.appendChild(dd);
            }
                    dt = document.createElement("dt");
                        label = document.createElement("label");
                        label.setAttribute("for", leftFrequencyRangeId_);
                        label.innerHTML = "Left Frequency: ";
                            input = document.createElement("input");
                            input.id = leftFrequencyDisplayId_;
                            input.setAttribute("value", "440");
                            input.setAttribute("style", "width: 40px");
                        label.appendChild(input);
                            span = document.createElement("span");
                            span.innerHTML = " Hz";
                        label.appendChild(span);
                    dt.appendChild(label);
                dl.appendChild(dt)
                    dd = document.createElement("dd");
                        input = document.createElement("input");
                        input.setAttribute("type", "range");
                        input.id = leftFrequencyRangeId_;
                        input.setAttribute("value", "440");
                        input.setAttribute("min", "20");
                        input.setAttribute("max", "1000");
                        input.setAttribute("step", "1");
                    dd.appendChild(input);
                dl.appendChild(dd);
                    dt = document.createElement("dt");
                        label = document.createElement("label");
                        label.setAttribute("for", rightFrequencyRangeId_);
                        label.innerHTML = "Right Frequency: ";
                            input = document.createElement("input");
                            input.id = rightFrequencyDisplayId_;
                            input.setAttribute("value", "430");
                            input.setAttribute("style", "width: 40px");
                        label.appendChild(input);
                            span = document.createElement("span");
                            span.innerHTML = " Hz";
                        label.appendChild(span);
                    dt.appendChild(label);
                dl.appendChild(dt)
                    dd = document.createElement("dd");
                        input = document.createElement("input");
                        input.setAttribute("type", "range");
                        input.id = rightFrequencyRangeId_;
                        input.setAttribute("value", "430");
                        input.setAttribute("min", "20");
                        input.setAttribute("max", "1000");
                        input.setAttribute("step", "1");
                    dd.appendChild(input);
                dl.appendChild(dd);
                    dt = document.createElement("dt");
                        label = document.createElement("label");
                        label.setAttribute("for", volumeRangeId_);
                        label.innerHTML = "Volume: ";
                            span = document.createElement("span");
                            span.id = volumeDisplayId_;
                            span.innerHTML = "1";
                        label.appendChild(span);
                    dt.appendChild(label);
                dl.appendChild(dt);
                    dd = document.createElement("dd");
                        input = document.createElement("input");
                        input.setAttribute("type", "range");
                        input.id = volumeRangeId_;
                        input.setAttribute("value", "1");
                        input.setAttribute("min", "0");
                        input.setAttribute("max", "1");
                        input.setAttribute("step", "0.05");
                    dd.appendChild(input);
                dl.appendChild(dd);
                    dt = document.createElement("dt");
                        label = document.createElement("label");
                        label.setAttribute("for", detuneRangeId_);
                        label.innerHTML = "Detune: ";
                            span = document.createElement("span");
                            span.id = detuneDisplayId_;
                            span.innerHTML = "0";
                        label.appendChild(span);
                            span = document.createElement("span");
                            span.innerHTML = " cent";
                        label.appendChild(span);
                    dt.appendChild(label);
                dl.appendChild(dt);
                    dd = document.createElement("dd");
                        input = document.createElement("input");
                        input.setAttribute("type", "range");
                        input.id = detuneRangeId_;
                        input.setAttribute("value", "0");
                        input.setAttribute("min", "-100");
                        input.setAttribute("max", "100");
                        input.setAttribute("step", "1");
                    dd.appendChild(input);
                dl.appendChild(dd);
                    dd = document.createElement("dd");
                        button = document.createElement("button");
                        button.id = startOrStopButtonId_;
                        button.setAttribute("type", "button");
                        button.setAttribute("class", "accel_brain_button");
                        button.innerHTML = "Play";
                    dd.appendChild(button);
                dl.appendChild(dd);
            fieldset.appendChild(dl);

            return fieldset;
        }
    };

}) ();

/**
 * Class of Composing dual object of BuildBeat for the Binaural Beat or Monaural Beat.
 *
 * @param {number} 0: Binaural Beat 1: Monaural Beat. 
 *
 */
var ComposeBeat = (function (){

    /** @private */
    var leftbuildBeatObj_ = false;
    /** @private */
    var rightbuildBeatObj_ = false;
    /** @private */
    var isPlaying_ = false;
    /** @private */
    var startOrStopButtonId_ = ComposeForm.startOrStopButtonId;
    /** @private */
    var volumeRangeId_ = ComposeForm.volumeRangeId;
    /** @private */
    var volumeDisplayId_ = ComposeForm.volumeDisplayId;
    /** @private */
    var waveFormFormId_ = ComposeForm.waveFormFormId;
    /** @private */
    var waveTypeRadioId_ = ComposeForm.waveTypeRadioId;
    /** @private */
    var leftFrequencyRangeId_ = ComposeForm.leftFrequencyRangeId;
    /** @private */
    var rightFrequencyRangeId_ = ComposeForm.rightFrequencyRangeId;
    /** @private */
    var leftFrequencyDisplayId_ = ComposeForm.leftFrequencyDisplayId;
    /** @private */
    var rightFrequencyDisplayId_ = ComposeForm.rightFrequencyDisplayId;
    /** @private */
    var detuneRangeId_ = ComposeForm.detuneRangeId;
    /** @private */
    var detuneDisplayId_ = ComposeForm.detuneDisplayId;
    /** @private */
    var beatTypeFormId_ = ComposeForm.beatTypeFormId;
    /** @private */
    var beatTypeRangeId_ = ComposeForm.beatTypeRangeId;

    /**
     * Handle an event derived by tuning volume.
     *
     * @private
     */
    function createControlVolumeRange_ () {
        document.getElementById(volumeRangeId_).addEventListener('input', function() {
            leftbuildBeatObj_.controlVolume(this.valueAsNumber);
            rightbuildBeatObj_.controlVolume(this.valueAsNumber);
            document.getElementById(volumeDisplayId_).textContent = this.value;
        }, false);
        document.getElementById(startOrStopButtonId_).addEventListener("click", function() {
            volume = document.getElementById(volumeRangeId_).valueAsNumber;
            leftbuildBeatObj_.controlVolume(volume);
            rightbuildBeatObj_.controlVolume(volume);
        }, false);
    }

    /**
     * Handle an event derived by selecting the wave type.
     *
     * @private
     */
    function createWaveTypeRange_ () {
        if (document.getElementById(waveFormFormId_)) {
            document.getElementById(waveFormFormId_).addEventListener('change', function() {
                for (var i = 0, len = this.elements[waveTypeRadioId_].length; i < len; i++) {
                    if (this.elements[waveTypeRadioId_][i].checked) {
                        type = this.elements[waveTypeRadioId_][i].value;
                        leftbuildBeatObj_.selectWaveType(type);
                        rightbuildBeatObj_.selectWaveType(type);
                        break;
                    }
                }
            }, false);
        }

        document.getElementById(startOrStopButtonId_).addEventListener("click", function() {
            if (document.getElementById(waveFormFormId_)) {
                waveType = document.getElementById(waveFormFormId_);
                for (var i = 0, len = waveType.elements[waveTypeRadioId_].length; i < len; i++) {
                    if (waveType.elements[waveTypeRadioId_][i].checked) {
                        type = waveType.elements[waveTypeRadioId_][i].value;
                        leftbuildBeatObj_.selectWaveType(type);
                        rightbuildBeatObj_.selectWaveType(type);
                        break;
                    }
                }
            } else {
                type = "sine";
                leftbuildBeatObj_.selectWaveType(type);
                rightbuildBeatObj_.selectWaveType(type);
            }
        }, false);
    }

    /**
     * Handle an event derived by tuning frequency.
     *
     * @private
     */
    function createFrequencyRange_ () {
        document.getElementById(leftFrequencyRangeId_).addEventListener('input', function() {
            frequency = this.valueAsNumber;
            leftbuildBeatObj_.controlFrequency(frequency);
            document.getElementById(leftFrequencyDisplayId_).value = this.value;
        }, false);

        document.getElementById(leftFrequencyDisplayId_).addEventListener('blur', function() {
            frequency = document.getElementById(leftFrequencyDisplayId_).value;
            leftbuildBeatObj_.controlFrequency(frequency);
            document.getElementById(leftFrequencyRangeId_).value = frequency;
        }, false);

        document.getElementById(rightFrequencyRangeId_).addEventListener('input', function() {
            frequency = this.valueAsNumber;
            rightbuildBeatObj_.controlFrequency(frequency);
            document.getElementById(rightFrequencyDisplayId_).value = this.value;
        }, false);

        document.getElementById(rightFrequencyDisplayId_).addEventListener('blur', function() {
            frequency = document.getElementById(rightFrequencyDisplayId_).value;
            rightbuildBeatObj_.controlFrequency(frequency);
            document.getElementById(rightFrequencyRangeId_).value = frequency;
        }, false);

        document.getElementById(startOrStopButtonId_).addEventListener("click", function() {
            frequency = document.getElementById(leftFrequencyRangeId_).valueAsNumber;
            leftbuildBeatObj_.controlFrequency(frequency);
        }, false);

        document.getElementById(startOrStopButtonId_).addEventListener("click", function() {
            frequency = document.getElementById(rightFrequencyRangeId_).valueAsNumber;
            rightbuildBeatObj_.controlFrequency(frequency);
        }, false);
    }

    /**
     * Handle an event derived by tuning detune.
     *
     * @private
     */
    function createDetuneRange_ () {
        document.getElementById(detuneRangeId_).addEventListener('input', function() {
            detune = this.valueAsNumber;
            leftbuildBeatObj_.controlDetune(detune);
            rightbuildBeatObj_.controlDetune(detune);
            document.getElementById(detuneDisplayId_).textContent = this.value;
        }, false);
        document.getElementById(startOrStopButtonId_).addEventListener("click", function() {
            detune = document.getElementById(detuneRangeId_).valueAsNumber;
            leftbuildBeatObj_.controlDetune(detune);
            rightbuildBeatObj_.controlDetune(detune);
        }, false);
    }

    /**
     * Handle an event derived by selecting beat type.
     *
     * @private
     */
    function createBeatType_ () {
        document.getElementById(beatTypeFormId_).addEventListener('change', function() {
            for (var i = 0, len = this.elements[beatTypeRangeId_].length; i < len; i++) {
                if (this.elements[beatTypeRangeId_][i].checked) {
                    type = this.elements[beatTypeRangeId_][i].value;
                    mode = type - 0;
                    switchMode_ (mode);
                    break;
                }
            }
        }, false);
    }

    /**
     * Handle an event derived by pushing the start or stop button.
     *
     * @private
     */
    function createStartOrStopButton_ () {
        document.getElementById(startOrStopButtonId_).addEventListener("click", function() {
            if (isPlaying_ == false) {
                leftbuildBeatObj_.playBeat();
                rightbuildBeatObj_.playBeat();
                isPlaying_ = true;
                this.innerHTML = '<span>Pause</span>';
            } else {
                leftbuildBeatObj_.stopBeat();
                rightbuildBeatObj_.stopBeat();
                isPlaying_ = false;
                this.innerHTML = '<span>Play</span>';
            }
        }, false);
    }

    /**
     * Setup event handlers.
     */
    function createDOMContent_ () {
        createStartOrStopButton_ ();
        createControlVolumeRange_ ();
        createWaveTypeRange_ ();
        createFrequencyRange_ ();
        createDetuneRange_ ();
        createBeatType_ ();
    }

    /**
     * Select the beat type. Switch one to another.
     *
     * @private
     * @param {number} 0: Binaraul Beat 1: Monaural Beat
     */
    function switchMode_ (mode) {
        stopFlag = false;
        if (leftbuildBeatObj_ != false && leftbuildBeatObj_.isPlaying ()) {
            leftbuildBeatObj_.stopBeat ();
            stopFlag = true;
        }

        if (rightbuildBeatObj_ != false && rightbuildBeatObj_.isPlaying()) {
            rightbuildBeatObj_.stopBeat ();
            stopFlag = true;
        }

        if (stopFlag) {
            isPlaying_ = false;
            this.innerHTML = '<span class="icon-start">Start</span>';
        }

        if (mode == 0) {
            leftbuildBeatObj_ = new BuildBeat(0);
            rightbuildBeatObj_ = new BuildBeat(1);
        }
        if (mode == 1) {
            leftbuildBeatObj_ = new BuildBeat(2);
            rightbuildBeatObj_ = new BuildBeat(2);
        }
    }

    /**
     * Select beat type.
     *
     * @constructor
     * @param {number} 0: Binaraul Beat 1: Monaural Beat
     */
    var constructor = function(mode) {
        if (mode == 0 || mode == 1) {
            switchMode_ (mode);
            if ((document.readyState === 'interactive') || (document.readyState === 'complete')) {
                createDOMContent_ ();
            } else {
                document.addEventListener('DOMContentLoaded', createDOMContent_, true);
            }

        } else {
            console.log("The value of LR must be 0 or 1 or 2.");
            return;
        }
    };

    /** @constructor */
    constructor.prototype = {
        /**
         * Select the beat type. Switch one to another.
         *
         * @param {number} 0: Binaraul Beat 1: Monaural Beat
         */
        switchMode: function (mode) {
            switchMode_ (mode);
        }
    };

    return constructor;

}) ();
