jQuery(document).ready(function(){
    $("#confirm").on("click", function(){
        const input = $("#in").val().trim()
        const mode = $('#mode').find(":selected").val();
        const data = {
            "text": input,
            "mode": mode
        }
        $("#loading").removeClass("visually-hidden")
        fetch("http://localhost:5050/annonymize", {
            method: "POST",
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json'
             // 'Content-Type': 'application/x-www-form-urlencoded',
        },
        })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                $("#loading").addClass("visually-hidden")
                $("#res").text(data.text)
            })
    })
})