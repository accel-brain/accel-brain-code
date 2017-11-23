# Subliminal perception

These JavaScript are tool for experimentation of subliminal perception.

## Description

This is a demo code for my case study in the context of my website.

### Usecase and code sample: Mere Exposure Effect

[js/mereexposure.js](https://github.com/chimera0/accel-brain-code/blob/master/Subliminal-Perception/js/mereexposure.js) is JavaScript library that enable to design the mere exposure effect. For instance, you can set words to be presented, color code of the word, and background color code during presentation. Stimulus presentation interval(milli seconds) can be also tuned.

```html
<script src='js/jquery-1.12.4.js'></script>
<script src='js/mereexposure.js'></script>
<script type="text/javascript">
    $(document).ready(function(){
        $("a").on("click", function(event) {
            // Set words to be presented.
            var allKeywordList = [
                "Singularity",
                "Transhumanism",
                "Artificial intelligence",
                "Aesthetics",
                "Semantik",
                "Rhythmus",
                "Katastrophe",
                "Aura",
                "Reiz schutz",
                "Surrealismus",
                "Gestalt",
                "Ursprung"
            ];
    
            // random choice.
            keywordList = [];
            for (i=0;i<5;i++)
            {
                var key = Math.round(Math.random() * allKeywordList.length) | 0;
                keywordList.push(allKeywordList.shift());
            }
                              
            // Initial setting.
            me = new MereExposure({
                "keywordList": keywordList, // words to be presented.
                "intervalMiliSec": 500, // Stimulus presentation interval(milli seconds).
                "backgroundColorCode": "#000", // Background color code during presentation.
                "backgroundOpacity": 1.0, // Make background color transparent.
                "backgroundImage": null, // Background Image. (aligin: center, middle)
                "colorCode": "#555", // Color code of words to be presented.
                "minFontSize": 18, // Minimum value of words to be presented.
                "maxFontSize": 45, // Maximum value of words to be presented.
                "deleteAllOther": false, // Delete all other dom objects ot not.
                "resetBgSheet": false, // Reset background sheet or not.
                "minTop": 0, // Minimum value of top.
            }).effect ();
        });
    });
</script>
```
#### Dependencies

- JQuery: v1.12.4 or higher.

#### More detail demos

- [Accel Brain; Beat](https://beat.accel-brain.com/)
    - The Subthreshold stimulation (so-called mere exposure effect) is to activate when you click any anchor link.

### Usecase and code sample: Link2Keyword

After loading [js/Link2Keyword.js](https://github.com/chimera0/accel-brain-code/blob/master/Subliminal-Perception/js/link2keyword.js), your anchor link click event brings about the page transition and smooth scrolling to the position of highlight keyword. You can tune the scroll speed.

```html
<html>
    <head>
        <script type="text/javascript" src="jquery-1.12.2.min.js"></script>
        <script type="text/javascript" src="jquery.highlight-5.js"></script>
        <script type="text/javascript" src="link2keyword.js"></script>
        <script type="text/javascript">
            $(document).ready(function() {
                Link2Keyword.init({
                    "contentArea": "accel_brain_area", 
                    "scrollSpeed": 800
                });
                Link2Keyword.highlight();
                Link2Keyword.scrolling();
            });
        </script>
        <style>
            .highlight {
                background-color: yellow;
            }
        </style>
    </head>
    <body>
        <div id="accel_brain_area">
            <ul>
                <li>
                    <a href="http://media.accel-brain.com/transhumanism-and-communication-with-ai/" target="_blank">my blog(Japanese)</a>
                </li>
            </ul>
        </div>
    </body>
</html>
```

#### Dependencies

- JQuery: v1.12.4 or higher.
- [jquery.highlight](http://johannburkard.de/blog/programming/javascript/highlight-javascript-text-higlighting-jquery-plugin.html): v5.

### More detail demos

- [Twitter](https://media.accel-brain.com/agency-operation-chimera0-2017-10-08-autotweety-net-connect-php/#link2keyword=Twitter) (Japanese)
    - Click here.

### Related PoC

- [ヴァーチャルリアリティにおける動物的「身体」の物神崇拝的なユースケース](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/) (Japanese)
    - [プロトタイプの開発：閾下知覚のメディア](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/2/#i-5)

## Version
- 1.0.1

## Author

- chimera0(RUM)

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0
