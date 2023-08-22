const min = 1;
const max = 11;


function randomIntFromInterval(min, max) { // min and max included 
    return Math.floor(Math.random() * (max - min + 1) + min)
}

const rndInt = randomIntFromInterval(min, max)
console.log("rndInts:")
console.log(rndInt)

// Instant redirect for DEBUG
//window.location.replace("interruption.html")

function delay(time) {
    return new Promise(resolve => setTimeout(resolve, time));
}

// Uncomment for Demo Mode: HTTP redirect after 10 seconds
//delay(10*1000).then(() => window.location.replace("interruption.html"));
// Uncoment for Experiment Mode: HTTP redirect after random amount of minutes (between min and max minutes)

function getIsDemo(){   
    return window.fs.getIsDemo();                    
}

getIsDemo().then((isDemo) => {
if (isDemo){
    console.log("DEMO 10 Seconds")
    delay(10*1000).then(() => window.location.replace("interruption.html"));
} else {
    delay(rndInt*60*1000).then(() => window.location.replace("interruption.html"));
}
})
