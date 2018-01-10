$('#file-upload').bind('change', function () {
    var path = '';
    path = $(this).val();
    $('#file-selected').html(path.split('\\').pop());
});

// Run on page load
window.onload = function () {
    // If sessionStorage is storing default values (ex. name), exit the function and do not restore data
    if (sessionStorage.getItem('mode') == "SVM") {
        return;
    }

    var mode = sessionStorage.getItem('mode');
    if (mode !== null) $('#mode').val(mode);

};

// Before refreshing the page, save the form data to sessionStorage
window.onbeforeunload = function () {
    sessionStorage.se//tItem("file", $('#file-upload').val());
    sessionStorage.setItem("mode", $('#mode').val());
};