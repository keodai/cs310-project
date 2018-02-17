$(document).ready(function() {
    $("#submit").click(function(event){
        progressBar();
    });
});

var expanded = 'false';

$("input:file").change(function () {
    var filenames = '';
    for (var i = 0; i < this.files.length; i++) {
        filenames += '<li>' + this.files[i].name + '</li>';
    }
    $('#file-selected').html('<ul>' + filenames + '</ul>');
});

// Run on page load
window.onload = function () {
    var mode = sessionStorage.getItem('mode');
    if (mode !== null) $('#mode').val(mode);
    var features = sessionStorage.getItem('features');
    if (features !== null) $('#features').val(features);
    var isExpanded = sessionStorage.getItem('expanded');
    if (isExpanded === 'true') {
        showOptions()
    } else {
        hideOptions()
    }
};

// Before refreshing the page, save the form data to sessionStorage
window.onbeforeunload = function () {
    sessionStorage.setItem("mode", $('#mode').val());
    sessionStorage.setItem("features", $('#features').val());
    sessionStorage.setItem("expanded", expanded);
};

function toggle(t, src) {
    if (t.innerText === 'Play') {
        t.innerText = 'Pause';
        $(t).parent('label').addClass('fa-stop').removeClass('fa-play');
        document.getElementById(src).play();
    } else {
        t.innerText = 'Play';
        document.getElementById(src).pause();
        $(t).parent('label').addClass('fa-play').removeClass('fa-stop');
    }
}

function uploadCheck() {
    if (document.getElementById('file').checked) {
        document.getElementById('file-input').style.display = 'block';
        document.getElementById('mic-selected').style.display = 'none';
    } else if (document.getElementById('mic').checked) {
        document.getElementById('file-input').style.display = 'none';
        document.getElementById('mic-selected').style.display = 'block';
    }
}

function showOptions() {
    document.getElementById('options-toggle').style.display = 'none';
    document.getElementById('options').style.display = 'block';
    expanded = 'true';
}

function hideOptions() {
    document.getElementById('options-toggle').style.display = 'block';
    document.getElementById('options').style.display = 'none';
    expanded = 'false';
}

$('#save-link').click(function () {
    var retContent = [];
    var retString = '';
    $('tbody tr').each(function (idx, elem) {
        var elemText = [];
        $(elem).children('td').each(function (childIdx, childElem) {
            elemText.push($(childElem).text().replace(/,/g, ""));
        });
        retContent.push(`${elemText.join(',')}`);
    });
    retString = retContent.join('\r\n');
    var file = new Blob([retString], {type: 'text/plain'});
    var btn = $('#save-link');
    btn.attr("href", URL.createObjectURL(file));
    btn.prop("download", "recommendations.txt");
});

function progressBar() {
    var method = document.forms["input-form"]["input-method"].value;
    var $progress = $('#progress');
    var $loading = $('#loading');
    var $vs = $('.vertical-scroll');

    if (method === 'mic') {
        $progress.css('display', 'block');
        $vs.css('display', 'none');
            setTimeout(function () {
                $progress.css('display', 'none');
                $loading.css('display', 'block');
            }, 5000); // Wait 5 seconds
    } else {
        $loading.css('display', 'block');
        $vs.css('display', 'none');
    }

}

