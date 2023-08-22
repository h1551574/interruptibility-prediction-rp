# Interruption App
## Option 1: Compile and Run the Interruption App
### Prerequisites
- Node.js: v18.17.1
- Electron-Forge: v6.4.0

#### Install Node.js
https://nodejs.org/download/release/v18.17.1

#### Install Electron Forge
Navigate to the "/Interruption App" directory and execute the following code to install electron-forge.
~~~
npm install --save-dev @electron-forge/cli@6.4.0
~~~
### Run Interruption App
To run the Interruption App simply navigate to the "/Interruption App" directory and execute the following code to run the app:
~~~
npm start
~~~

### Build Interruption App
To build an executable for Windows (.exe) run:
~~~
npm run make
~~~

You will find the .exe file unde "/out/make/squirrel.windows/x64"

This portable file can be run on any Windows system.

## Option 2: Run the Precompiled Executable
You will find the "interruption-app-2.0.0 Setup.exe" executable und "/Releases". Simply run this file on a Windows system. This method does not require any prerequisites or compilation steps.