const pasteArea = document.getElementsByClassName("pasteArea")[0];
const upload = document.getElementsByClassName("upload")[0];
const resultsContainer = document.getElementsByClassName("resultsContainer")[0];
const loadingIcon = document.getElementsByClassName("loading")[0];
const searchButton = document.getElementsByClassName("searchButton")[0];
const popUpContainer = document.getElementsByClassName("popUpContainer")[0];
const saveBox = document.getElementsByClassName("saveBox")[0];

// Empty text of paste aria when clicked
pasteArea.addEventListener("focus", function(){
    if(pasteArea.value == "Or paste text here..."){
        pasteArea.value = "";
    }
});

// Search for copied text after clicking search button
searchButton.addEventListener("click", function(){
    // Check if not empty
    if(pasteArea.value.length <= 50){
        // Make border red
        pasteArea.classList.add("pasteIncorrect");
        return;
    }
    // Create file from text
    var file = new File([pasteArea.value], "upload.txt");

    search(file)
});

// Save checkbox
saveBox.addEventListener('change', function(){
    if(saveBox.checked){
        popUpContainer.style.display = "flex";
        setTimeout(function(){
            popUpContainer.style.opacity = 1;
        }, 10)
    }
});

// Pop up
function popUpAccept(){
    popUpContainer.style.opacity = 0;
    setTimeout(function(){
        popUpContainer.style.display = "none";
    }, 500);
}

function popUpReject(){
    saveBox.checked = false;
    popUpContainer.style.opacity = 0;
    setTimeout(function(){
        popUpContainer.style.display = "none";
    }, 500);
}

// *****FILE UPLOAD

// By clicking
upload.addEventListener("click", onClickImage);
function onClickImage(){
    let input = document.createElement('input');
    input.type = 'file';
    input.accept = '.txt';
    
    input.onchange = _ => {
        imageFile = Array.from(input.files)[0];
        search(imageFile)
        //previewImage();
        input.remove();
    };
    input.style.display = 'none';
    document.body.appendChild(input);
    input.click();
}

//Drop image
//Prevent default behavior
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    upload.addEventListener(eventName, preventDefaults, false);
});
function preventDefaults (e) {
    e.preventDefault();
    e.stopPropagation();
}

// Change the look of the box when you hover with a file
['dragenter', 'dragover'].forEach(eventName => {
    upload.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    upload.addEventListener(eventName, unhighlight, false);
});
  
function highlight(e) {
    upload.classList.add("highlight");
}

function unhighlight(e) {
    upload.classList.remove("highlight");
}

// Handle drop
upload.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
  let dt = e.dataTransfer;
  imageFile = dt.files[0];
  // Wait small amount of time for file to be uploaded
  setTimeout(function(){
    search(imageFile)
  }, 500)
}

function search(file){
    // Clear error border if shown
    pasteArea.classList.remove("pasteIncorrect");
    // Clear previous output if available
    [...resultsContainer.children].forEach(child => {
        if(child.classList.contains("result")){
            resultsContainer.removeChild(child);
        }
    });
    // Move to the results
    moveToResults();
    // Make call to backend
    const request = new XMLHttpRequest();
    request.open('POST', 'search', true);
    var formdata = new FormData();
    formdata.append("file", file);

    request.onreadystatechange = function(){
        if(this.readyState == 4){
            if(this.status == 200){
                // Hide loading symbol
                loadingIcon.style.display = "none";
                loadingIcon.style.opacity = 0;
                // Parse json data
                var resultJson = JSON.parse(this.responseText);
                for(var row = 1; row < resultJson.length; row++){
                    createResult(resultJson[row][1], resultJson[row][2], resultJson[row][3], resultJson[row][4]);
                }
                // Fade in results
                showResults();
            } else {
                alert("Oops something went wrong while we tried to connect to the server...");
            }
        }
    }

    request.send(formdata);
}

function createResult(filename, sim, scaleText, previewText){
    // Create result container
    var result = document.createElement("div");
    result.classList.add("result");

    // Create result header
    var rHeader = document.createElement("div");
    rHeader.classList.add("resultHeader");

    // Find document title and format
    var extensionIndex = filename.lastIndexOf('.');
    var titleText = filename.substring(0, extensionIndex);
    var formatText = filename.substring(extensionIndex);

    // Create text in header
    var title = document.createElement("span");
    title.classList.add("resultTitle");
    title.innerHTML = titleText;
    rHeader.appendChild(title);

    var format = document.createElement("span");
    format.classList.add("resultFormat");
    format.innerHTML = formatText;
    rHeader.appendChild(format);

    var label  = document.createElement("span");
    label.classList.add("resultLabel");
    label.innerHTML = "Similarity:";
    rHeader.appendChild(label);

    var similarity = document.createElement("span");
    similarity.classList.add("resultSimilarity");
    similarity.innerHTML = sim;
    rHeader.appendChild(similarity);

    var scale = document.createElement("span");
    scale.classList.add("resultScale");
    scale.innerHTML = scaleText;
    if(scaleText == "high"){
        scale.style.backgroundColor = "#31b800";
    } else if(scaleText == "medium"){
        scale.style.backgroundColor = "#006e99";
    } else if(scaleText == "low"){
        scale.style.backgroundColor = "#f2352e";
    }
    if(parseFloat(sim) > 0.99){
        scale.innerHTML = "searched document";
        scale.style.backgroundColor = "#f29d2e";
    }
    rHeader.appendChild(scale);
    result.appendChild(rHeader);

    // Create preview
    var preview = document.createElement("span");
    preview.classList.add("resultPreview");
    preview.innerHTML = previewText;
    result.appendChild(preview);

    // Add result to container
    resultsContainer.appendChild(result);
}

// Animations
function moveToResults(){
    //Show loading icon
    loadingIcon.style.display = "block";
    setTimeout(function(){
        loadingIcon.style.opacity = 1;
    }, 10);
    window.scrollTo({
        top: resultsContainer.offsetTop - 20,
        left: 0,
        behavior: 'smooth'
      });
}

function showResults(){
    var counter = 0;
    [...resultsContainer.children].forEach(child => {
        if(child.classList.contains("result")){
            setTimeout(function(){
                child.style.opacity = 1;
            }, counter);
            counter += 50;
        }
    });
}

