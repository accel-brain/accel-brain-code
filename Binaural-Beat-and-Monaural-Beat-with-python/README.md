# Binaural Beats and Monaural Beats with Python

`AccelBrainBeat` is a Python library for creating the binaural beats or monaural beats. You can play these beats and generate wav files. The frequencys can be optionally selected.

## Description

This Python script enables you to handle your mind state by a kind of "Brain-Wave Controller" which is generally known as Biaural beat or Monauarl beats in a simplified method.

## Documentation

Full documentation is available on [https://code.accel-brain.com/Binaural-Beat-and-Monaural-Beat-with-python/](https://code.accel-brain.com/Binaural-Beat-and-Monaural-Beat-with-python/) . This document contains information on functionally reusability, functional scalability and functional extensibility.

## Demonstration IN Movie

- [Drive to design the brain's level upper](https://www.youtube.com/channel/UCvQNSr2fVjI8bIMhJ_bfQmg) (Youtube)

## Installation

Install using pip:

```bash
pip install AccelBrainBeat
```

### Source code

The source code is currently hosted on GitHub.

- [Binaural-Beat-and-Monaural-Beat-with-python](https://github.com/chimera0/accel-brain-code/tree/master/Binaural-Beat-and-Monaural-Beat-with-python)

### Python package index(PyPI)

Binary installers for the latest released version are available at the Python package index.

- [AccelBrainBeat: Python Package Index](https://pypi.python.org/pypi/AccelBrainBeat/)

### Dependencies

- [NumPy](http://www.numpy.org/): v1.7.0 or higher

#### To play the beats on console

If you want to not only output wav files but also play the beats on console, [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) (v0.2.9 or higher) must be installed.

## Use-case on console

You can study or work while listening to the Binaural or Monauarl beats. Before starting your job, run a batch program on console.

### Create "Binaural Beat" and output wav file

Run the batch program: [save_binaural_beat.py](https://github.com/chimera0/accel-brain-code/blob/master/Binaural-Beat-and-Monaural-Beat-with-python/bat/save_binaural_beat.py).

```bash
python bat/save_binaural_beat.py -o binaural_beat.wav -l 400 -r 430 -t 60 -v 0.01
```

The command line arguments is as follows.

```bash
python bat/save_binaural_beat.py -h
```
```
usage: save_binaural_beat.py [-h] [-o OUTPUT_FILE_NAME] [-l LEFT] [-r RIGHT]
                             [-t TIME] [-v VOLUME]

Create the Binaural Beat and save wav file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE_NAME, --output_file_name OUTPUT_FILE_NAME
                        Output file name.
  -l LEFT, --left LEFT  Left frequencys (Hz).
  -r RIGHT, --right RIGHT
                        Right frequencys (Hz).
  -t TIME, --time TIME  Play time. This is per seconds.
  -v VOLUME, --volume VOLUME
                        Sound volume.
```

### Create "Monaural Beat" and output wav file

Run the batch program: [save_monaural_beat.py](https://github.com/chimera0/accel-brain-code/blob/master/Binaural-Beat-and-Monaural-Beat-with-python/bat/save_monaural_beat.py).

```bash
python bat/save_monaural_beat.py -o monaural_beat.wav -l 400 -r 430 -t 60 -v 0.01
```

The command line arguments is as follows.

```bash
python bat/save_monaural_beat.py -h
```
```
usage: save_monaural_beat.py [-h] [-o OUTPUT_FILE_NAME] [-l LEFT] [-r RIGHT]
                             [-t TIME] [-v VOLUME]

Create the Monaural Beat and save wav file.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FILE_NAME, --output_file_name OUTPUT_FILE_NAME
                        Output file name.
  -l LEFT, --left LEFT  Left frequencys (Hz).
  -r RIGHT, --right RIGHT
                        Right frequencys (Hz).
  -t TIME, --time TIME  Play time. This is per seconds.
  -v VOLUME, --volume VOLUME
                        Sound volume.
```

### Create and play "Binaural Beat" on console

Run the batch program: [play_binaural_beat.py](https://github.com/chimera0/accel-brain-code/blob/master/Binaural-Beat-and-Monaural-Beat-with-python/bat/play_binaural_beat.py).

```bash
python play_binaural_beat.py -l 400 -r 430 -t 60 -v 0.01
```

The command line arguments is as follows.

```bash
python bat/play_binaural_beat.py -h
```

```
usage: play_binaural_beat.py [-h] [-l LEFT] [-r RIGHT] [-t TIME] [-v VOLUME]

Create and play the Binaural Beat.

optional arguments:
  -h, --help            show this help message and exit
  -l LEFT, --left LEFT  Left frequencys (Hz).
  -r RIGHT, --right RIGHT
                        Right frequencys (Hz).
  -t TIME, --time TIME  Play time. This is per seconds.
  -v VOLUME, --volume VOLUME
                        Sound volume.
```

### Create and play "Monaural Beat" on console

Run the batch program: [play_monaural_beat.py](https://github.com/chimera0/accel-brain-code/blob/master/Binaural-Beat-and-Monaural-Beat-with-python/bat/play_monaural_beat.py).

```bash
python bat/play_monaural_beat_beat.py -l 400 -r 430 -t 60 -v 0.01
```

The command line arguments is as follows.

```bash
python bat/play_monaural_beat.py -h
```
```
usage: play_monaural_beat.py [-h] [-l LEFT] [-r RIGHT] [-t TIME] [-v VOLUME]

Create and play the Monaural Beat.

optional arguments:
  -h, --help            show this help message and exit
  -l LEFT, --left LEFT  Left frequencys (Hz).
  -r RIGHT, --right RIGHT
                        Right frequencys (Hz).
  -t TIME, --time TIME  Play time. This is per seconds.
  -v VOLUME, --volume VOLUME
                        Sound volume.
```

## Use-case for coding

You can use this library as a module by executing an import statement in your Python source file.

### Create wav file of "Binaural Beat"

Call the method.

```python
from AccelBrainBeat.brainbeat.binaural_beat import BinauralBeat

brain_beat = BinauralBeat() # for binaural beats.
brain_beat.save_beat(
    output_file_name="save_binaural_beat.wav",
    frequencys=(400, 430),
    play_time=10,
    volume=0.01
)
```

- `output_file_name` is wav file name or path.

### Create wav file of "Monaural Beat"

The interface of monaural beats is also same as the binaural beats.

```python
from AccelBrainBeat.brainbeat.monaural_beat import MonauralBeat

brain_beat = MonauralBeat() # for monaural beats.
brain_beat.save_beat(
    output_file_name="save_monaural_beat.wav",
    frequencys=(400, 430),
    play_time=10,
    volume=0.01
)
```

### Create and play "Binaural Beat"

For example, if `400` Hz was played in left ear and `430` Hz in the right, then the binaural beats would have a frequency of 30 Hz.

Import Python and Cython modules.

```python
from AccelBrainBeat.brainbeat.binaural_beat import BinauralBeat
```

Instantiate objects and call the method.

```python
brain_beat = BinauralBeat()

brain_beat.play_beat(
    frequencys=(400, 430),
    play_time=10,
    volume=0.01
)
```

- The type of `frequencys` is tuple. This is a pair of both frequencys.
- `play_time` is playing times(per seconds).
- `volume` is the sound volume. It depends on your environment.

### Create and play "Monaural Beat"

The interface of monaural beats is same as the binaural beats. `MonoauralBeat` is functionally equivalent to `BinauralBeat`.

```python
from AccelBrainBeat.brainbeat.monaural_beat import MonauralBeat

brain_beat = MonauralBeat()

brain_beat.play_beat(
    frequencys=(400, 430),
    play_time=10,
    volume=0.01
)
```

## Licence

- [GPL2](https://github.com/chimera0/Binaural-Beat-and-Monaural-Beat-with-python/blob/master/LICENSE)

## Related products

 Binaural beats and Monauarl beats can be implemented by not only Python but also Unity3D. I developed Unity3D package: [Immersive Brain's Level Upper by Binaural Beat and Monaural Beat.](https://www.assetstore.unity3d.com/en/#!/content/66518).

 As the kind of "Brain-Wave Controller", this Unity3D package is functionally equivalent to Python`s library.


## More detail

 The function of this library is inducing you to be extreme immersive mind state on the path to peak performance. You can handle your mind state by using this library which is able to control your brain waves by the binaural beats and the monaural beats.

### Concept of Binaural beats and Monauarl beats

 According to a popular theory, brain waves such as Delta, Theta, Alpha, Beta, and Gamma rhythms tend to be correlated with mind states. The delta waves(1-3 Hz) are regarded as the slowest brain waves that are typically produced during the deep stages of sleep. The theta waves(4-7 Hz) are offen induced by the meditative state or focusing the mind. The alpha waves(8-12 Hz) are associate with relaxed state. The beta waves(13-29 Hz) are normal waking consciousness. The Gamma waves(30-100 Hz) are the fastest of the brain waves and associated with peak concentration and the brain's optimal frequency for cognitive functioning.

 By a theory of the binaural beats, signals of two different frequencies from headphone or earphone are presented separately, one to each ear, your brain detects the phase variation between the frequencies and tries to reconcile that difference. The effect on the brain waves depends on the difference in frequencies of each tone. For example, if 400 Hz was played in one ear and 430 in the other, then the binaural beats would have a frequency of 30 Hz.

 The monaural beats are similar to the binaural beats. But they vary in distinct ways. The binaural beats seem to be "created" or perceived by cortical areas combining the two different frequencies. On the other hand, the monaural beats are due to direct stimulation of the basilar membrane. This makes it possible to hear the beats.

 Please choose either binaural beets or monaural beats. If you set up 5 Hz, your brain waves and the frequency can be tuned and then you are able to be the meditative state or focusing the mind. Or what you choose to be relaxed state is the alpha waves(8-12 Hz).

### Related PoC

- [仏教の社会構造とマインドフルネス瞑想の意味論](https://accel-brain.com/social-structure-of-buddhism-and-semantics-of-mindfulness-meditation/) (Japanese)
    - [プロトタイプの開発：バイノーラルビート](https://accel-brain.com/social-structure-of-buddhism-and-semantics-of-mindfulness-meditation/3/#i-6)

## Author

- chimera0(RUM)

### Author's websites

- [Accel Brain](https://accel-brain.com) (Japanese)

### References

- Brandy, Queen., et al., (2003) “Binaural Beat Induced Theta EEG Activity and Hypnotic Susceptibility : Contradictory Results and Technical Considerations,” American Journal of Clinical Hypnosis, pp295-309.
- Green, Barry., Gallwey, W. Timothy., (1986) The Inner Game of Music, Doubleday.
- Kennerly, Richard Cauley., (1994) An empirical investigation into the effect of beta frequency binaural beat audio signals on four measures of human memory, Department of Psychology, West Georgia College, Carrolton, Georgia.
- Kim, Jeansok J., Lee, Hongjoo J., Han, Jung-Soo., Packard, Mark G. (2001) “Amygdala Is Critical for Stress-Induced Modulation of Hippocampal Long-Term Potentiation and Learning,” The Journal of Neuroscience, Vol. 21, pp5222-5228.
- LeDoux, Joseph. (1998) The emotional brain : the mysterious underpinnings of emotional life, London : Weidenfeld & Nicolson.
- McEwen, Bruce S., Sapolsky, Robert M. (1995) “Stress and cognitive function,” Current Opinion in Neurobiology, Vol. 5, pp205-216.
- Oster, Gerald., (1973) “Auditory Beats in the Brain,” Scientific American, pp94-102.
- Radford, Benjamin., (2001) “Pokemon Contagion: Photosensitive Epilepsy or Mass Psychogenic Illness?,” Southern Medical Journal, Vol. 94, No. 2, pp197-204.
- Steward, Oswald., (2000) Functional neuroscience, Springer.
- Swann, R., et al. (1982) The Brain ? A User’s Manual, New York: G. P. Putnam’s Sons.
- Takeo, Takahashi., et al., (1999) “Pokemon seizures,” Neurol J Southeast Asia, Vol. 4, pp1-11.
- Vollenweider., Franz X., Geyer., Mark A. (2001) “A systems model of altered consciousness: Integrating natural and drug-induced psychoses,” Brain Research Bulletin, Vol. 56, No. 5, pp495-507.
- Wahbeh, Helane., Calabrese, Carlo., Zwickey, Heather., (2007) “Binaural Beat Technology in Humans : A Pilot Study to Assess Psychologic and Physiologic Effects,” The Journal of Alternative and Complementary Medicine, Vol. 13, No. 1, pp25-32.
- Westman, Jack C., Walters, James R. (1981) “Noise and Stress : A Comprehensive Approach,” Environmental Health Perspectives, Vol. 41, pp291-309.
