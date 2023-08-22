const interruptionStart = new Date;
var interruptionLag = null;



var calculationForm = document.getElementById("calculation-form")
calculationForm.style.display = "none";

var calculationResult = document.getElementById("calculation-result")

var surveyDiv = document.getElementById("survey-div")
surveyDiv.style.display = "none";

const min = 10;
const max = 99;

function randomIntFromInterval(min, max) { // min and max included 
    return Math.floor(Math.random() * (max - min + 1) + min)
}

const randomnumber1 = randomIntFromInterval(min, max)
const randomnumber2 = randomIntFromInterval(min, max)
const randomMuliplication = randomnumber1*randomnumber2;

var calculationLabel = document.getElementById("calculation-label")
calculationLabel.innerHTML = `What is ${randomnumber1}x${randomnumber2}?`


var calculationStartButton = document.getElementById("calculation-start-button")

var calculationSubmitButton = document.getElementById("calculation-submit-button")
calculationSubmitButton.style.display = "none";

calculationStartButton.addEventListener("click", function() {
    interruptionLag = Date.now() - interruptionStart;
    calculationForm.style.display = "block";
    calculationSubmitButton.style.display = "block";
    calculationStartButton.style.display = "none";
})


window.addEventListener("DOMContentLoaded", function() {
    document.getElementById('calculation-form').addEventListener("submit", function(e) {
        e.preventDefault(); // before the code
        /* do what you want with the form */
        
        // Should be triggered on form submit
        var calculationAnswer = document.getElementById("calculation").value
        calculationForm.style.display = "none";
        calculationSubmitButton.style.display = "none";
        calculationResult.innerHTML = `Your answer was: ${calculationAnswer} The correct answer is: ${randomMuliplication}`

        surveyDiv.style.display = "block";
    })
  });

// calculationSubmitButton.addEventListener("click", function() {
//     var calculationAnswer = document.getElementById("calculation").value

//     if(calculationAnswer==""){
//         alert("Calculation must be filled out!")
//     } else {
//         calculationForm.style.display = "none";
//         calculationSubmitButton.style.display = "none";
//         calculationResult.innerHTML = `Your answer was: ${calculationAnswer} The correct answer is: ${randomMuliplication}`

//         surveyDiv.style.display = "block";
//     }
// })

function getDisableHeadrestWarning(){   
    return window.fs.getDisableHeadrestWarning();                    
}

function getIsDemo(){   
    return window.fs.getIsDemo();                    
}

var surveySubmitButton = document.getElementById("survey-submit-button")
surveySubmitButton.addEventListener("click", function() {
     // TODO:
    const data = {
        tsStart: interruptionStart,
        tsEnd: new Date,
        interruption_lag: interruptionLag,
        disturbance: document.querySelector('input[name="disturbance"]:checked').value,
        interruptibility: document.querySelector('input[name="interruptibility"]:checked').value,
        mental_workload: document.querySelector('input[name="mental-workload"]:checked').value
    }
    
    console.log(JSON.stringify(data))

    getIsDemo().then((isDemo) => {
        if(isDemo){
            window.fs.appendData(data)
        }
    })
    getDisableHeadrestWarning().then((disableHeadrestWarning) => {
        console.log(disableHeadrestWarning)
        if(disableHeadrestWarning){
            window.location.replace("black.html")
        } else {
            window.location.replace("headrest.html")
        }
    })
})






