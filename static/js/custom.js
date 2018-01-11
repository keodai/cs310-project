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
    sessionStorage.se//tItem("file", $('#file-upload').val());
    sessionStorage.setItem("mode", $('#mode').val());
};