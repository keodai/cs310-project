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