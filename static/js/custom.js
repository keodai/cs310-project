$(document).ready(function() {
  // $(".submit").click(function() {
  //   // $(".submit").addClass("loading");
  //
  // })
    $("#submit").click(function(event){
        progressBar();
    });
});

// $('#input-form').ajaxForm(function(response) {
//   return response
// });



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

};

// Before refreshing the page, save the form data to sessionStorage
window.onbeforeunload = function () {
    sessionStorage.setItem("mode", $('#mode').val());
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
}

function hideOptions() {
    document.getElementById('options-toggle').style.display = 'block';
    document.getElementById('options').style.display = 'none';
}

$('#save-link').click(function () {
    var retContent = [];
    var retString = '';
    $('tbody tr').each(function (idx, elem) {
        var elemText = [];
        $(elem).children('td').each(function (childIdx, childElem) {
            elemText.push($(childElem).text());
        });
        retContent.push(`(${elemText.join(',')})`);
    });
    retString = retContent.join(',\r\n');
    var file = new Blob([retString], {type: 'text/plain'});
    var btn = $('#save-link');
    btn.attr("href", URL.createObjectURL(file));
    btn.prop("download", "recommendations.txt");
});

function progressBar() {
    var method = document.forms["input-form"]["input-method"].value;
    var $progress = $('#progress');
    // var $progressBar = $('.progress-bar');
    var $loading = $('#loading');

    if (method === 'mic') {
        $progress.css('display', 'block');

        // setTimeout(function () {
        //     $progressBar.css('width', '20%');
        //     setTimeout(function () {
        //         $progressBar.css('width', '40%');
        //         setTimeout(function () {
        //             $progressBar.css('width', '60%');
        //             setTimeout(function () {
        //                 $progressBar.css('width', '80%');
        //                 setTimeout(function () {
        //                     $progressBar.css('width', '100%');
                            setTimeout(function () {
                                $progress.css('display', 'none');
                                $loading.css('display', 'block');
                            }, 5000); // WAIT 1 second
        //                 }, 1000); // WAIT 1 second
        //             }, 1000); // WAIT 1 second
        //         }, 1000); // WAIT 1 second
        //     }, 1000); // WAIT 1 second
        // }, 1000); // WAIT 1 second
    } else {
        $loading.css('display', 'block');
    }

}

