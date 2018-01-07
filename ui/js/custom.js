$("#run_btn").click(function () {
    $.getJSON('http://127.0.0.1:5000/recommend?src=x&mode=y',
        function (data, textStatus, jqXHR) {
            alert(data);
        }
    )

});

$("#test").click(function () {
    $.getJSON('http://127.0.0.1:5000/test',
        function (data, textStatus, jqXHR) {
            document.getElementById("output").innerHTML = data.result;
            //alert(data.result);
            //$( "#output" ).html(data);
        }
    )
//     $.ajax({
//    url: 'http://127.0.0.1:5000/test',
//    success: function(response) {
//      // here you do whatever you want with the response variable
//        alert(response);
//    }
// });

});


$('#file-upload').bind('change', function () {
    var path = '';
    path = $(this).val();
    $('#file-selected').html(path.split('\\').pop());
})