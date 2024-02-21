var videoContainerList = document.getElementsByClassName("video-compare-container")
var videoClipperList = document.getElementsByClassName("video-clipper")
var video1ready = new Array(videoContainerList.length).fill(false);
var video2ready = new Array(videoContainerList.length).fill(false);

for (var i = 0; i < videoContainerList.length; i++) {
    var videoContainer = videoContainerList[i]
    var videoClipper = videoClipperList[i]

    videoContainer.addEventListener("mousemove", trackLocation(videoContainer, videoClipper));
    videoContainer.addEventListener("touchstart", trackLocation(videoContainer, videoClipper));
    videoContainer.addEventListener("touchmove", trackLocation(videoContainer, videoClipper));

    var video1 = videoClipper.getElementsByTagName("video")[0]
    var video2 = videoContainer.getElementsByTagName("video")[0]

    // Check canplay event to know when both videos are ready to play and start them.

    video1.addEventListener('canplay', canplay1(video1, video2, i));
    video2.addEventListener('canplay', canplay2(video1, video2, i));
    // Stopped to load next frame, make the other video wait too.
    video1.addEventListener('waiting', () => { video2.pause(); });
    video2.addEventListener('waiting', () => { video1.pause(); });
    // Resumed from buffering, continue playing the other video too.
    video1.addEventListener('playing', () => { video2.play(); });
    video2.addEventListener('playing', () => { video1.play(); });
}

function tryStart(video1, video2, video1ready, video2ready) {
    if (video1ready && video2ready) {
        video1.play();
        video2.play();
    }
}

function canplay1(video1, video2, i) {
    return function () {
        video1ready[i] = true;
        tryStart(video1, video2, video1ready[i], video2ready[i]);
    }
}

function canplay2(video1, video2, i) {
    return function () {
        video2ready[i] = true;
        tryStart(video1, video2, video1ready[i], video2ready[i]);
    }
}


function trackLocation(videoContainer, videoClipper) {
    return function (e) {
        var clippedVideo = videoClipper.getElementsByTagName("video")[0]
        var rect = videoContainer.getBoundingClientRect()
        var position = ((e.pageX - rect.left) / videoContainer.offsetWidth) * 100

        if (position <= 100) {
            videoClipper.style.width = position + "%";
            clippedVideo.style.width = ((100 / position) * 100) + "%";
            clippedVideo.style.zIndex = 3;
        }
    }
}