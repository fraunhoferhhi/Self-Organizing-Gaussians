```bash
ffmpeg -i path1/rotate.mp4 -i path2/rotate.mp4 -filter_complex hstack=inputs=2 output.mp4 -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -movflags faststart -crf 18
```