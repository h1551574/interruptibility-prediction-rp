const { app, BrowserWindow, ipcMain } = require('electron')
const path = require('path')
const fs = require('fs');

const dataPath = './Data';

var dataSetName = 'test-no-date.csv';
var isDemoConf = false;
var disableHeadrestWarningConf = false;

if (require('electron-squirrel-startup')) app.quit();



const getIsDemo = () => {
  return isDemoConf
}

const setIsDemo = (_,isDemo) => {
  isDemoConf = isDemo
}

const getDisableHeadrestWarning = () => {
  return disableHeadrestWarningConf
}

const setDisableHeadrestWarning = (_,disableHeadrestWarning) => {
  disableHeadrestWarningConf = disableHeadrestWarning
}

const makeNewDataSet = (_,id) => {
  const headers = 'tsStart, tsEnd, interruption_lag, disturbance, interruptibility, mental_workload\n'

  try {
    if (!fs.existsSync(dataPath)) {
      fs.mkdirSync(dataPath);
    }
  } catch (err) {
    console.error(err);
  }

  const experimentStart = Date.now();

  // optional participant
  if (id == ''){
    dataSetName = `test-${experimentStart}.csv`
  } else {
    dataSetName = `test-participant${id}-${experimentStart}.csv`
  }

  fs.writeFile(path.join(dataPath, dataSetName), headers, err => {
    if (err) {
      console.error(err);
    }
    // file written successfully
  });
}

const appendData = (_, data) => {
  if(!isDemoConf){
    //const dataString = JSON.stringify(data)
    const dataString = `${data.tsStart.toISOString()}, ${data.tsEnd.toISOString()}, ${data.interruption_lag}, ${data.disturbance}, ${data.interruptibility}, ${data.mental_workload}`
    
    try {
      if (!fs.existsSync(dataPath)) {
        fs.mkdirSync(dataPath);
      }
    } catch (err) {
      console.error(err);
    }
    
    fs.appendFile(path.join(dataPath, dataSetName), dataString+"\n", err => {
      if (err) {
        console.error(err);
      }
      // file written successfully
    });
  }
}

const createWindow = () => {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    autoHideMenuBar: true,
    webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
    },
  });
  ipcMain.handle('ping', () => 'pong')
  ipcMain.handle('getIsDemo',getIsDemo)
  ipcMain.handle('getDisableHeadrestWarning',getDisableHeadrestWarning)
  win.loadFile('index.html');
  ipcMain.on('makeNewDataSet',makeNewDataSet)
  ipcMain.on('appendData',appendData)
  ipcMain.on('setIsDemo',setIsDemo)
  ipcMain.on('setDisableHeadrestWarning',setDisableHeadrestWarning)
};

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
