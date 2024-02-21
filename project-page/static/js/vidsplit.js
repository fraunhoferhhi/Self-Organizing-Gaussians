document.addEventListener('DOMContentLoaded', function () {
    const video = document.getElementsByClassName('sourceVideo')[0];
    const canvas = document.getElementsByClassName('videoCanvas')[0];
    const ctx = canvas.getContext('2d');


    let splitX = canvas.width; // Initial split position

    video.addEventListener('loadedmetadata', () => {
        // Adjust canvas size based on the video frame, halving the height for display
        canvas.width = video.videoWidth / 2;
        canvas.height = video.videoHeight;
    });

    video.play();

    function draw() {
        if (video.paused || video.ended) return;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw the left part of the top video
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

        // Draw the right part of the bottom video
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
        ctx.beginPath(); // Begin a new path for the line
        ctx.moveTo(splitX, 0); // Move to the start point of the line at the top of the canvas
        ctx.lineTo(splitX, canvas.height); // Draw a line to the bottom of the canvas
        ctx.strokeStyle = 'white'; // Set the color of the line
        ctx.lineWidth = 1; // Set the line width
        ctx.stroke(); // Render the line
        
        requestAnimationFrame(draw);
    }

    video.onplay = () => {
        draw();
    };

    canvas.addEventListener('mousemove', function (e) {
        const rect = canvas.getBoundingClientRect();
        splitX = e.clientX - rect.left;
        splitX = Math.max(0, Math.min(splitX, canvas.width));
        draw();
    });
});
