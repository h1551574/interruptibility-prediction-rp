var startButton = document.getElementById("start-button")
startButton.addEventListener("click", function() {
    const participantID = document.getElementById("participant-id").value
    const isDemo = document.getElementById("demo-mode").checked
    const disableHeadrestWarning = document.getElementById("disable-headrest-warning").checked
    window.fs.setIsDemo(isDemo)
    window.fs.setDisableHeadrestWarning(disableHeadrestWarning)
    if(!isDemo){
        window.fs.makeNewDataSet(participantID)
    }
    window.location.replace("black.html")
})


