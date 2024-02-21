let glob_splitX = []

document.addEventListener('DOMContentLoaded', function () {
    const videos = document.getElementsByClassName('sourceVideo');
    const canvases = document.getElementsByClassName('videoCanvas');
    const videoContainers = document.getElementsByClassName('video-container');

    for (let i = 0; i < videos.length; i++) {
        let cur_video = videos[i];
        let cur_canvas = canvases[i];
        let cur_ctx = cur_canvas.getContext('2d');
        let cur_video_container = videoContainers[i];
        glob_splitX.push(50);

        cur_video.addEventListener('loadedmetadata', load_event(cur_canvas, cur_video_container));
        cur_video.play();
        cur_video.addEventListener('playing', play_event(cur_video, cur_canvas, cur_ctx, i, cur_video_container));
        cur_canvas.addEventListener('mousemove', move_event(cur_canvas, i, cur_video_container))
    }
});

function load_event(canvas, video_container){
    return () => {
        console.log(video_container.offsetWidth)
        canvas.width = video_container.offsetWidth;
        canvas.height = video_container.offsetHeight;
    }
}

function play_event(video, canvas, ctx, index, video_container) {
    return () => {
        setInterval(() => {
            draw(video, canvas, ctx, glob_splitX[index], video_container);
        }, 5);
    }
}

function move_event(canvas, index){
    return (e) => {
        const rect = canvas.getBoundingClientRect();
        glob_splitX[index] = e.clientX - rect.left;
        glob_splitX[index] = Math.max(0, Math.min(glob_splitX[index], canvas.width));
    }
}

function draw(video, canvas, ctx, splitX, video_container) {
    if (video.paused || video.ended) return;
    if (ctx === undefined) return;
    // console.log(index, glob_splitX[index])

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    splitX_imgSpace = splitX * (0.5 * video.videoWidth) / video_container.offsetWidth
    ctx.drawImage(
        video,
        0, // sx
        0, // sy
        splitX_imgSpace, // sWidth
        video.videoHeight, // sHeight
        0, // dx
        0, // dy
        splitX, //dWidth
        canvas.height // dHeight
    );

    ctx.drawImage(
        video,

        video.videoWidth / 2 + splitX_imgSpace,
        0,
        video.videoWidth / 2 - splitX_imgSpace,
        video.videoHeight,

        splitX,
        0,
        canvas.width - splitX,
        canvas.height
    );

    // Draw a vertical line at the split
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
    requestAnimationFrame(draw);
}
