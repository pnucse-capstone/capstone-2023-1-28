/*!
    * Start Bootstrap - SB Admin v7.0.7 (https://startbootstrap.com/template/sb-admin)
    * Copyright 2013-2023 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-sb-admin/blob/master/LICENSE)
    */
    // 
// Scripts
//
// "small_button" id를 가진 <div> 요소에 클릭 이벤트 리스너를 추가하십시오.

window.addEventListener('DOMContentLoaded', event => {

    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        // Uncomment Below to persist sidebar toggle between refreshes
        // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
        //     document.body.classList.toggle('sb-sidenav-toggled');
        // }
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }

});

document.querySelectorAll('.dropdown-item').forEach(function (item) {
    item.addEventListener('click', function () {
        var selectedItemText = item.textContent;
        document.getElementById('selectedItemText').textContent = selectedItemText;
    });
});

// 인풋에 숫자 아니면 제거해뿌기
document.getElementById("thresh").addEventListener("input", function (e) {
    var inputValue = e.target.value;
    e.target.value = inputValue.replace(/[^0-9]/g, "");
});

document.getElementById("kernel").addEventListener("input", function (e) {
    var inputValue = e.target.value;
    e.target.value = inputValue.replace(/[^0-9]/g, "");
});

function changeCursor() {
    document.querySelector('[value]').style.cursor = 'pointer';
}


var imageDisplayTimer;

function showImages(button) {
    var imageContainer = button.parentElement.querySelector('.image-slide');
    if (imageContainer.classList.contains('hidden')) {
        imageContainer.classList.remove('hidden');
    } else {
        imageContainer.classList.add('hidden');
    }
}


function different() {
    var dropdown = document.getElementById("inputGroupSelect04");
    var selectedValue = dropdown.options[dropdown.selectedIndex].value;        // 드롭박스 value
    var threshValue = document.getElementById("thresh").value;
    var kernelValue = document.getElementById("kernel").value;

    if (threshValue === "") {
        threshValue = 801;
    }
    if (kernelValue === "") {
        kernelValue = 23;
    }



    var imagesToDisplay = [];
    var currentIndex = 0;
    var inputImages = [];
    var outputImages = [];

    // 이미지 표시 타이머 중지
    clearTimeout(imageDisplayTimer);

    // /predict 라우트에 대한 AJAX 요청을 보냅니다.
    $.ajax({
        url: '/prepro',
        method: 'POST',
        data: {
            selectValue: selectedValue,
            threshValue: threshValue,
            kernelValue: kernelValue
        },
        success: function(response) {
            imagesToDisplay = [];
            currentIndex = 0;
            // inputImages = [];
            outputImages = [];

            response.forEach(function(imagePath) {
                imagePath = imagePath.replace(/\\/g, '/');
                imagesToDisplay.push(imagePath);

                outputImages.push(imagePath);

                // if (imagePath.startsWith('static/results' + url_param + '/png/input_')) {
                //     inputImages.push(imagePath);
                // } else if (imagePath.startsWith('static/results' + url_param + '/png/output_')) {
                //     outputImages.push(imagePath);
                // }
            });
            console.log(outputImages)

            // 이미지 표시
            function displayImages() {
                if (currentIndex < outputImages.length) {
                    // var inputImage = document.createElement('img');
                    var outputImage = document.createElement('img');

                    // inputImage.src = inputImages[currentIndex];
                    outputImage.src = outputImages[currentIndex];
                    outputImage.style.width = '50%';
                    outputImage.style.display = 'block'; // 블록 레벨 요소로 설정
                    outputImage.style.margin = '0 auto'; // 가로 중앙 정렬

                    // html 내부에 미리 정해놓은 id로 넣기.
                    // $('#input_body').empty().append(inputImage);
                    $('#prepro_area').empty().append(outputImage);
                    currentIndex++;
                }
                // 4초 마다 refresh
                if (currentIndex < outputImages.length) {
                    imageDisplayTimer = setTimeout(displayImages, 4000);
                }
            }

            // 이미지 표시
            displayImages();
        }
    });
}


function predict(url_param) {
    var imagesToDisplay = [];  // 응답 이미지 배열
    var currentIndex = 0;  // 현재 표시 중인 이미지의 인덱스
    var inputImages = [];
    var outputImages = [];

    // 이미지 표시 타이머 중지
    clearTimeout(imageDisplayTimer);

    // /predict 라우트에 대한 AJAX 요청을 보냅니다.
    $.ajax({
        url: '/predict' + url_param,
        method: 'POST',
        success: function(response) {
            // 이미지 배열 초기화
            imagesToDisplay = [];
            currentIndex = 0;
            inputImages = [];
            outputImages = [];

            // 응답 이미지를 배열에 저장합니다.
            response.forEach(function(imagePath) {
                // 역슬래시를 슬래시로 교체합니다.
                imagePath = imagePath.replace(/\\/g, '/');
                imagesToDisplay.push(imagePath);

                // input과 output 이미지를 분류합니다.
                if (imagePath.startsWith('static/results' + url_param + '/png/input_')) {
                    inputImages.push(imagePath);
                } else if (imagePath.startsWith('static/contour' + url_param + '/png/output_')) {
                    outputImages.push(imagePath);
                }
            });

            // 이미지를 표시하는 함수 설정
            function displayImages() {
                if (currentIndex < inputImages.length) {
                    var inputImage = document.createElement('img');
                    var outputImage = document.createElement('img');

                    inputImage.src = inputImages[currentIndex];
                    outputImage.src = outputImages[currentIndex];

                    // 이미지를 각각의 div에 추가합니다.
                    $('#input_body').empty().append(inputImage);
                    $('#result_body').empty().append(outputImage);
                    currentIndex++;
                }
                if (currentIndex < inputImages.length) {
                    // 4초 후에 다음 짝의 이미지를 표시합니다.
                    imageDisplayTimer = setTimeout(displayImages, 4000);
                }
            }

            // 초기 이미지 표시를 시작합니다.
            displayImages();
        }
    });
}

function updateProgressBar(percent) {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = percent + '%';
}



function anomaly_detect(url_param) {
    var imagesToDisplay = [];  // 응답 이미지 배열
    var currentIndex = 0;  // 현재 표시 중인 이미지의 인덱스
    var inputImages = [];
    var outputImages = [];
    var originDir = 'static/temp' + url_param + '/';
    var inputDir = 'static/cutpaste/datasets' + url_param + '/';
    var outputDir = 'static/cutpaste/results' + url_param + '/';
    var tList = document.getElementById('datatablesSimple');
    var tBody = tList.getElementsByTagName('tbody')[0]


    // 이미지 표시 타이머 중지
    clearTimeout(imageDisplayTimer);

    // /predict 라우트에 대한 AJAX 요청을 보냅니다.
    $.ajax({
        url: '/anomaly' + url_param,
        method: 'POST',
        success: function(response) {
            // 이미지 배열 초기화
            imagesToDisplay = [];
            currentIndex = 0;
            inputImages = [];
            outputImages = [];

            function addList(filename, isNormal, url_param){
                var newRow = document.createElement('tr');
                var nameCell = document.createElement("td");
                var sizeCell = document.createElement("td");
                var anomalCell = document.createElement("td");
                var checkCell = document.createElement("td");
                var imgLink = document.createElement('div');
                var size = '';


                nameCell.textContent = filename;
                if( isNormal == 1 ){
                    isNormal = "정상";
                }else {
                    isNormal = "비정상";
                }
                anomalCell.textContent = isNormal;
                if (url_param == '/small') {
                    size = "소형관";
                }else if (url_param == '/middle') {
                    size = "중형관";
                }else {
                    size = '대형관';
                }
                sizeCell.textContent = size;

                imgLink.textContent = '확인하기';
                imgLink.classList.add('cursor_pointer');
                imgLink.onclick = function(){
                    clickImage(this);
                }
                imgLink.setAttribute('value1', inputDir + filename);
                imgLink.setAttribute('value2', originDir + filename);

                checkCell.appendChild(imgLink);

                newRow.appendChild(nameCell);
                newRow.appendChild(anomalCell);
                newRow.appendChild(sizeCell);
                newRow.appendChild(checkCell);

                tBody.appendChild(newRow);
                console.log(tBody);
                tBody.insertRow(newRow);
            }

            for(var filename in response){
                if (response.hasOwnProperty(filename)) {
                    var isNormal = response[filename];
                    filename = filename + '.png';
                    inputImages.push(inputDir + filename);
                    if ( isNormal == 1 ){
                        // outputImages.push("static/assets/img/normal.png");
                        outputImages.push("정상");
                    }else{
                        outputImages.push(outputDir + filename);
                    }
                    addList(filename, isNormal, url_param);
                }
            }

            // 이미지를 표시하는 함수 설정
            function displayImages() {
                if (currentIndex < inputImages.length) {
                    var inputImage = document.createElement('img');

                    inputImage.style.maxWidth='100%';
                    inputImage.style.maxHeight='100%';

                    inputImage.src = inputImages[currentIndex];

                    var outputValue = outputImages[currentIndex];
                    if (outputValue == '정상') {
                        var outputImage = document.createElement('span');
                        outputImage.textContent = "정상입니다.";
                    }else{
                        var outputImage = document.createElement('img');
                        outputImage.style.maxWidth='100%';
                        outputImage.style.maxHeight='100%';

                        outputImage.src = outputImages[currentIndex];
                    }
                    // 이미지를 각각의 div에 추가합니다.
                    $('#input_body').empty().append(inputImage);
                    $('#result_body').empty().append(outputImage);

                    currentIndex++;
                }
                if (currentIndex < inputImages.length) {
                    // 4초 후에 다음 짝의 이미지를 표시합니다.
                    imageDisplayTimer = setTimeout(displayImages, 4000);
                }
            }

            displayImages();
        }
    });
}

function clickImage(clickedElement) {
    var imagePath1 = clickedElement.getAttribute('value1');
    var imagePath2 = clickedElement.getAttribute('value2');

    var imageElement1 = document.createElement('img');
    imageElement1.style.maxWidth = '100%';
    imageElement1.style.maxHeight = '100%';
    imageElement1.src = imagePath1;

    var imageElement2 = document.createElement('img');
    imageElement2.style.maxWidth = '100%';
    imageElement2.style.maxHeight = '100%';
    imageElement2.src = imagePath2;

    var clickImgElement1 = document.getElementById('click_expand');
    clickImgElement1.innerHTML = ''; // 이전 이미지를 지웁니다.
    clickImgElement1.appendChild(imageElement1);

    var clickImgElement2 = document.getElementById('click_origin');
    clickImgElement2.innerHTML = ''; // 이전 이미지를 지웁니다.
    clickImgElement2.appendChild(imageElement2);


}
