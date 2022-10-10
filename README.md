<img src="./make-a-video.png" width="400px"></img>

## Make-A-Video - Pytorch (wip)

Implementation of <a href="https://makeavideo.studio/">Make-A-Video</a>, new SOTA text to video generator from Meta AI, in Pytorch. They combine pseudo-3d convolutions (axial convolutions) and temporal attention and show much better temporal fusion.

The pseudo-3d convolutions isn't a new concept. It has been explored before in other contexts, say for protein contact prediction as <a href="https://www.biorxiv.org/content/10.1101/2022.08.04.502748v2.full">"dimensional hybrid residual networks"</a>.

The gist of the paper comes down to, take a SOTA text-to-image model (here they use DALL-E2, but the same learning points would easily apply to Imagen), make a few minor modifications for <a href="https://arxiv.org/abs/2204.03458">attention across time</a> and other ways to skimp on the compute cost, do frame interpolation correctly, get a great video model out.

<a href="https://www.youtube.com/watch?v=AcvmyqGgMh8">AI Coffee Break explanation</a>

## Citations

```bibtex
@misc{Singer2022,
    author  = {Uriel Singer},
    url     = {https://makeavideo.studio/Make-A-Video.pdf}
}
```
