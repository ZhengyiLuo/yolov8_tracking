
paper.install(window);


send_action = function (action, query) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "http://172.29.225.72:8080/talk_server", true);
    var requestJson = JSON.stringify({
        "action": action,
        "query": query
    });
    xhr.send(requestJson);

    xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
        var response = xhr.responseText;
        // Do something with the response
        $.toast({message: 'Server says: ' + response})
    }
    };
    
}

get_request = function (query) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", `http://172.29.225.72:8080/${query}`, true);
    xhr.send( null );
    return xhr;
}
  
const y_scale = 20
const x_scale = 20
const cam_pos = [271.38562, 20.05252075]
const colorList = ["red", "blue", "yellow", "violet", "green", "orange", "pink", "brown",  "grey", "white", "purple", "teal" , "olive"]

window.onload = function() {
    console.log("here we go again!")

    $("#laguageMode").click(function(){
        $("#videoMode").addClass("grey");
        $("#videoMode").removeClass("green");
        $("#laguageMode").addClass("green");
        $("#laguageMode").removeClass("grey");
        send_action("set_mode", {"mode": "language"});
    })

    $("#videoMode").click(function(){
        $("#videoMode").addClass("green");
        $("#videoMode").removeClass("grey");
        $("#laguageMode").addClass("grey");
        $("#laguageMode").removeClass("green");
        send_action("set_mode", {"mode": "video"});
    })
    
    $("#sendPrompt").click(function(){
        console.log("sending prompt")
        send_action("set_prompt", {"prompt": $("#prompt").val()});
        $.toast({
            displayTime: 'auto',
            showProgress: 'top',
            classProgress: 'red',
            displayTime: 10000,
            message: 'Running for prompt: ' + $("#prompt").val(),
            });
    })

      $("#sendOffset").click(function(){
        console.log("sending offset")
        send_action("set_offset", {"offset": $("#offset").val()});
    })
}

