let files = [], // will be store images


button = document.querySelector('.topbutton'), // uupload button
form = document.querySelector('.form'), // form ( drag area )
dropzone = document.querySelector('.dropzone')
container = document.querySelector('.filelist'), // container in which image will be insert
container_heading = document.querySelector('.filesuploaded .heading')
text = document.querySelector('.text'), // inner text of form
browse = document.querySelector('.select'), // text option fto run input
input = document.querySelector('.file'); // file input

console.log(button);
console.log(form);
console.log(dropzone);
console.log(container);
console.log(container_heading);
console.log(text);
console.log(browse);
console.log(input);

browse.addEventListener('click', () => input.click());

// input change event
input.addEventListener('change', () => {
	let file = input.files;

	for (let i = 0; i < file.length; i++) {
		if (files.every(e => e.name !== file[i].name)) files.push(file[i])
	}

	//form.reset();
	showImages();
})

const showImages = () => {
	let images = '';
	files.forEach((e, i) => {

		images += `<div class="filename">
                        <span class="closebut" onclick="delImage(${i})">+</span>
                        <span class='nameoffile'>${e.name}</span>
                    </div>`
	})

    container_heading.innerHTML = 'Documents uploaded: ' + files.length.toString();
	container.innerHTML = images;
} 

const delImage = index => {
	files.splice(index, 1)
	showImages()
} 

// drag and drop 
dropzone.addEventListener('dragover', e => {
	e.preventDefault()

	dropzone.classList.add('dragover')
	text.innerHTML = 'Drop images here'
})

dropzone.addEventListener('dragleave', e => {
	e.preventDefault()

	dropzone.classList.remove('dragover')
	text.innerHTML = 'Drag & drop image here or <span class="select">Browse</span>'
})

dropzone.addEventListener('drop', e => {
	e.preventDefault()

    dropzone.classList.remove('dragover')
	text.innerHTML = 'Drag & drop image here or <span class="select">Browse</span>'

	let file = e.dataTransfer.files;
	for (let i = 0; i < file.length; i++) {
		if (files.every(e => e.name !== file[i].name)) files.push(file[i])
	}

	showImages();
})

button.addEventListener('click', () => {
	let form = new FormData();
	files.forEach((e, i) => form.append(`file[${i}]`, e))
    

	// now you can send the image to server using AJAX or FETCH API
	
})