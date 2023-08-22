const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron,
  ping: () => ipcRenderer.invoke('ping'),
  // we can also expose variables, not just functions
})

contextBridge.exposeInMainWorld('fs', {
  makeNewDataSet: (id) => ipcRenderer.send('makeNewDataSet',id),
  appendData: (data) => ipcRenderer.send('appendData',data),
  getIsDemo: () => ipcRenderer.invoke('getIsDemo'),
  setIsDemo: (isDemo) => ipcRenderer.send('setIsDemo',isDemo),
  getDisableHeadrestWarning: () => ipcRenderer.invoke('getDisableHeadrestWarning'),
  setDisableHeadrestWarning: (disableHeadrestWarning) => ipcRenderer.send('setDisableHeadrestWarning',disableHeadrestWarning),
  // we can also expose variables, not just functions
})