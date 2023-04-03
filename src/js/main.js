
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
        $.toast({message: 'Server says: ' + response, displayTime: 10000,})
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
    let record_toggle = false
    let cycle_motion_toggle = true

    $("#record").click(function(){
        console.log("!")
        record_toggle = !record_toggle
        if (record_toggle){
            send_action("start_record", {});
            $("#record").addClass("red");
            $("#record").removeClass("grey");
            $("#record").text("Recording")
        } else {
            send_action("end_record", {});
            $("#record").addClass("grey");
            $("#record").removeClass("red");
            $("#record").text("Not Recording")
            
        }
    })

    $("#cycleMotion").click(function(){
        console.log("!!!")
        cycle_motion_toggle = !cycle_motion_toggle
        if (cycle_motion_toggle){
            send_action("set_cycle_motion", {"cycle_motion": cycle_motion_toggle});
            $("#cycleMotion").addClass("green");
            $("#cycleMotion").removeClass("grey");
            $("#cycleMotion").text("Cycle Motion")
        } else {
            send_action("set_cycle_motion", {"cycle_motion": cycle_motion_toggle});
            $("#cycleMotion").addClass("grey");
            $("#cycleMotion").removeClass("green");
            $("#cycleMotion").text("No Cylce Motion")
            
        }
    })

    $("#reset").click(function(){
        send_action("reset", {});
    })

    $("#setDefaultPose").click(function(){
        send_action("set_default_pose", {});
        
    })

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
    })

    $("#sendOffset").click(function(){
        console.log("sending offset")
        send_action("set_offset", {"offset": $("#offset").val()});
    })

    $("#sendBuffer").click(function(){
        console.log("sending buffer")
        send_action("set_buffer", {"buffer": $("#buffer").val()});
    })
    
}

