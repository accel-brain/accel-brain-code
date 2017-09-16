# Subliminal perception

## Description
- This is a demo code for my case study in the context of my website.

### Code sample

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

### More detail demos

- [Accel Brain; Beat](https://beat.accel-brain.com/)
    - The Subthreshold stimulation (so-called mere exposure effect) is to activate when you click any anchor link.

### Related Case Studies

- [ヴァーチャル・リアリティにおける動物的「身体」の蒐集を媒介としたサイボーグ的な物神崇拝](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/)
    - [ケーススタディ：閾下知覚のメディア](https://accel-brain.com/cyborg-fetischismus-in-sammlung-von-animalisch-korper-in-virtual-reality/2/#i-5)

## Version
- 1.0.1

## Author

- chimera0

## Author URI

- http://accel-brain.com/

## License

- GNU General Public License v2.0
