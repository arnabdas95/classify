// wait for the content of the window element
// to load, then performs the operations.
// This is considered best practice.
document.getElementById("text").style.display = "none";
document.getElementById("text_two").style.display = "none";

 let xyposition=[];
 let i =0;
 let minx=miny=300;
 let maxx=maxy=0;
window.addEventListener('load', ()=>{


	resize(); // Resizes the canvas once the window loads
	document.addEventListener('mousedown', startPainting);
	document.addEventListener("touchstart",startPainting);
	document.addEventListener('mouseup', stopPainting);
	document.addEventListener("touchend",stopPainting);
	document.addEventListener('mousemove', sketch);
	document.addEventListener("touchmove",sketch);
	window.addEventListener('resize', resize);
});

const canvas = document.querySelector('#myCanvas');

// Context for the canvas for 2 dimensional operations
const ctx = canvas.getContext('2d');

// Resizes the canvas to the available size of the window.
function resize(){
ctx.canvas.width = 300;
ctx.canvas.height =300;
}

// Stores the initial position of the cursor
let coord = {x:0 , y:0};

// This is the flag that we are going to use to
// trigger drawing
let paint = false;

// Updates the coordianates of the cursor when
// an event e is triggered to the coordinates where
// the said event is triggered.
function getPosition(event){
coord.x = event.clientX - canvas.offsetLeft;
coord.y = event.clientY - canvas.offsetTop;
}

// The following functions toggle the flag to start
// and stop drawing
function startPainting(event){
paint = true;
getPosition(event);
}
function stopPainting(){
paint = false;
}

function sketch(event){
if (!paint) return;
ctx.beginPath();

ctx.lineWidth = 20;

// Sets the end of the lines drawn
// to a round shape.
ctx.lineCap = 'round';

ctx.strokeStyle = 'black';

// The cursor to start drawing
// moves to this coordinate
ctx.moveTo(coord.x, coord.y);

// The position of the cursor
// gets updated as we move the
// mouse around.
getPosition(event);

// A line is traced from start
// coordinate to this coordinate
ctx.lineTo(coord.x , coord.y);
//store x and y coordinate


let posx=coord.x;
let posy=coord.y;
if (posx>maxx){maxx=posx;}
if (posy>maxy){maxy=posy;}
if (posx<minx){minx=posx;}
if( posy<miny){miny=posy;}


//instead of ctx.lineWidth/2 use only width as I want to padding some area
xyposition=[minx-ctx.lineWidth,miny-ctx.lineWidth,maxx+ctx.lineWidth,maxy+ctx.lineWidth];

// Draws the line.
ctx.stroke();
}



 function submitForm() {
     //document.getElementById("show_result").style.display='block';
    var pixel = document.getElementById('myCanvas');

    var img_pixel_data = pixel.toDataURL();
 document.getElementById('text').value=img_pixel_data;
 document.getElementById('text_two').value=xyposition;
 console.log(xyposition);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
        document.getElementById("form").submit();
    }



document.getElementById('submit').onclick = function() {
       submitForm();
}
//function submitForm(e){
//    e.preventDefault();
//    const formData = new FormData(); // Currently empty
//
////    let img_pixel_data=ctx.getImageData(0,0,300,300);
////    img_pixel_data = img_pixel_data['data'];
////
//
//var pixel = document.getElementById('myCanvas');
//var img_pixel_data = pixel.toDataURL();




//
//    console.log(e);
//    formData.set('text',img_pixel_data);
//     //document.getElementById("clear").click();
//    //formData.set('canvas_image',img_pixel_data);
//     ctx.clearRect(0, 0, canvas.width, canvas.height);
//    fetch('/predict',{method:'POST',body:formData})
//
//
//}
//





