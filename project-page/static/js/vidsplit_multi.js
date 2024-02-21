let glob_splitX = []

document.addEventListener('DOMContentLoaded', function () {
    const videos = document.getElementsByClassName('sourceVideo');
    const canvases = document.getElementsByClassName('videoCanvas');
    for (let i = 0; i < videos.length; i++) {
        let cur_video = videos[i];
        let cur_canvas = canvases[i];
        let cur_ctx = cur_canvas.getContext('2d');
        glob_splitX.push(50);

        cur_video.addEventListener('loadedmetadata', load_event(cur_canvas, cur_video));
        cur_video.play();
        cur_video.addEventListener('playing', play_event(cur_video, cur_canvas, cur_ctx, i));
        cur_canvas.addEventListener('mousemove', move_event(cur_canvas, i))
    }
});

function load_event(canvas, video){
    return () => {
        canvas.width = video.videoWidth / 2;
        canvas.height = video.videoHeight;
    }
}

function play_event(video, canvas, ctx, index) {
    return () => {
        setInterval(() => {
            draw(video, canvas, ctx, glob_splitX[index]);
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

function draw(video, canvas, ctx, splitX) {
    if (video.paused || video.ended) return;
    if (ctx === undefined) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(
        video,
        0, // sx
        0, // sy
        splitX, // sWidth
        video.videoHeight, // sHeight
        0, // dx
        0, // dy
        splitX, //dWidth
        video.videoHeight // dHeight
    );

    ctx.drawImage(
        video,
        video.videoWidth / 2 + splitX,
        0,
        video.videoWidth / 2 - splitX,
        video.videoHeight,
        splitX,
        0,
        video.videoWidth / 2 - splitX,
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
