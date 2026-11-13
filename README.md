# Un-Mix-a-Lot for ComfyUI (Beta)

> **Beta:** this is an early version. It works in our tests, but it has not been tried on many setups yet. Bug reports are welcome in the Issues tab.

Three ComfyUI nodes that split audio into **voices**, **music** and **sound effects**, split voices into one track per speaker, and mix the pieces back together. Useful for cleaning up dialogue, removing music from a clip, or rebuilding the soundtrack of an AI video. It is the engine from the free [Un-Mix-a-Lot app](https://github.com/aharonrabinowitz-create/un-mix-a-lot), as nodes.

## Use at your own risk

Free software provided **"as is", with no warranty of any kind**. Separation is never perfect and the nodes may have bugs. The author(s) and contributors are not responsible for lost or damaged files or any other harm. Always work on copies of your files. Full text: [DISCLAIMER.txt](DISCLAIMER.txt).

## Install

**With ComfyUI-Manager:** Install via Git URL, paste this repo's address, then restart ComfyUI.

**By hand:** open a terminal in `ComfyUI/custom_nodes` and run

```
git clone https://github.com/aharonrabinowitz-create/comfyui-un-mix-a-lot.git
```

then restart ComfyUI. There is nothing else to install: the nodes use the PyTorch and torchaudio that ComfyUI already has.

The first time you run the node it downloads the BandIt v2 model (about 447 MB) into `ComfyUI/models/un_mix_a_lot/`. After that it works offline.

## The nodes

Find them under **audio / Un-Mix-a-Lot**.

**Un-Mix: Voices / Music / Effects (BandIt v2)**

- Input: `audio` (any ComfyUI AUDIO, mono or stereo, any sample rate).
- Outputs: `voices`, `music`, `effects` (AUDIO, same length, channel count and sample rate as the input).
- `device`: auto, cpu or cuda. It works on a CPU (in our test on a basic 2-core machine, 12 seconds of audio took about 24 seconds) and is much faster on a GPU.
- `keep_model_loaded`: keep the model in memory for faster repeat runs. Off by default so it frees GPU memory for the rest of your workflow.
- Long audio is handled in chunks with crossfades. Internally the audio is mixed to mono at 48 kHz, so the three outputs are mono signals copied to every channel.

**Un-Mix: Split voices (optional)**

You only need this if you want one track per speaker. Skip it otherwise.

- Input: `audio` (a voices track, for example the `voices` output above) and `num_voices` (2 to 4).
- Outputs: `voice_1` to `voice_4` (AUDIO, the same length and format as the input) and `timeline` (text). Voices you didn't ask for come out silent. The voice tracks always add back up to the original exactly.
- It listens for who is talking and groups the speech by voice. Voice 1 is whoever speaks first.
- Works best with different people or different AI voices. One person doing several voices is hard to tell apart and may be grouped wrongly. People talking at the same time end up on one track.
- **Fixing mistakes:** connect the `timeline` output to a `Preview Any` node to read it, for example `0.00-4.50 voice 1`. Copy it into the `corrections` box on this node, change what is wrong (for instance move a time, or change `voice 2` to `voice 3`), and run again. If the box has text, the node uses your version instead of guessing. Leave it empty for the automatic result. For tricky material you can also split it in the Un-Mix-a-Lot app, fix it there with the timeline editor, and bring the finished tracks into ComfyUI.
- The first time you run it, it installs three small helper libraries (SpeechBrain, Silero VAD and SciPy) into ComfyUI's Python and downloads an 80 MB voice model. That takes about a minute and needs internet once.

**Un-Mix: Mix stems**

- Inputs: `voices`, `music`, `effects` (all optional) and a gain for each (0 to 2). Set a gain to 0 to remove that part, or turn one up or down.
- `prevent_clipping`: scales the result down if it would go over full volume.

### Example: remove the music from a video's audio

`Load Audio` (or the audio from a loaded video) -> **Un-Mix: Voices / Music / Effects** -> **Un-Mix: Mix stems** with `music_gain` at 0 -> `Save Audio`.

You can also send `voices`, `music` and `effects` straight to separate `Save Audio` nodes to get all three as files.

## Notes

- The three pieces add back up to roughly the original sound, not exactly. In our test the difference was a few percent.
- Quality depends on the material. Dialogue over music and effects works best. Singing, whispers, and sounds that mimic speech can end up in the wrong piece.
- Built and tested with ComfyUI on a CPU with PyTorch 2.11. Not yet tested on every GPU or ComfyUI version.

## Licenses and credits

- These nodes: MIT (see [LICENSE](LICENSE)).
- **Voice splitting** uses [SpeechBrain](https://speechbrain.github.io/)'s ECAPA speaker model (Apache-2.0) and [Silero VAD](https://github.com/snakers4/silero-vad) (MIT), installed on first use.
- **BandIt v2** by Karn Watcharasupat et al. (Apache-2.0; the code is included in `bandit_v2/` with its license). The model weights are [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) (Zenodo record 12701995) and are downloaded on first use, not bundled. If you share audio made with them, credit the BandIt v2 authors.
- Not affiliated with or endorsed by ComfyUI, the BandIt authors or any other project named here. All names are trademarks of their owners.
