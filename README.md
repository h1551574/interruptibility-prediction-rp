# replication-package

**Overview**
The replication package consists of six main directories:
  1. Data Processing
  2. Data
  3. Experiment Documents
  4. Interruption App
  5. JHotDraw
  6. Result Analysis

**Data Processing: Overview**
Data Processing includes the following sub-directories:

  1. ”Data Pre-Processing”: all the Python source code used for pre-processing(as described in Section 5.2.7)
  2. ”Feature Selection and Model Building”: protocol to replicate the allmanual steps in the Weka GUI as well as the Java source code for the automated steps (as described in Section 5.2.8)
  3. Python source code for calculating the correlations used in the result analysis of Section 7.3
  4. Python source code for the descriptive statistics used in Section 7.1.

**Data Processing**: Pre-Processing The Data Pre-Processing directory contains two Python scripts. Firstly, ”Baseline Feature Calculation” needs to be run first to calculate the baseline features used for the normalization step described in Section 5.2.7. Secondly, ”Feature Calculation and Normalization” is run to calculate the features from the experiment sessions and
normalize them using the previously calculated baseline features. While the former is a simple Python script, the latter is a Jupyter Notebook containing additional documentation embedded as Markdown text.

**Dependency Management** The ”python-requirements.txt” file provides a list of dependencies with exact version, which can be used by pip to install all packages necessary to run the Python code in the ”Data Processing” directory. The Java code is built as a Maven project and includes a pom.xml file which specifies all dependencies with exact versions. The manual steps in the machine learning procedure require Weka 3.8.0 installed on your machine.

**Data** The Data directory includes the interruption data. Due to data privacy, the sensor data cannot be shared in the replication package. ECG data might be especially sensitive since it could be used to derive information about the participants’ health (e.g. heart problems). Simple pseudonymization would not have sufficed to enable the sharing of this data set, as pseudonymized data is still treated as personal data in the GDPR [76]. The interruption data does not include personal data, as such it will be shared in the replication package.

**Experiment Documents** The ”Experiment Documents” directory contains documents which are used during the experiment session itself:
  1. Two versions of the experiment protocol with detailed instructions for the experimenter
     1. Full version: used in the replication
     2. Reduced version: excluding elements added due to extension
  2. Pre-Interview questionnaire
  3. Post-Interview questionnaire
  4. Document explaining sensor setup
  5. YouTube link to fish tank movie used for baseline recording
  6. PowerPoint Presentation used for task explanation and demo run of simulated interruptions
  7. Document explaining hexagon geometry, which can be used by the participants during the programming task.

**Interruption App**
The "Interruption App" directory contains the source code of the rebuilt tablet app for simulating interruption during the experiment. It contains a short tutorial of how to run and compile the code. Alternatively, a precompiled executable file is also contained in the "Releases" directory, which does not require any compilation steps.

**JHotDraw**
The "JHotDraw 7.0.6" directory contains two versions of the project code used by the participants in the programming session (for more details on project and tasks, see Section 4.3.1); 1) a "Clean" version without any changes made, to be used as a starting point for the programming task, and 2) a "Solved" version, were the code change tasks have been implemented, to be used as a reference for the experimenters. Additionally, this directory contains the two icons ("createCircle.png" and "createHexagon.png") which can be used by the participant during the programming tasks.

**Results Analysis**
The "Results Analysis" directory contains all spreadsheets used for the analysis of the results used in Section 7  as well as the evaluation of the pre- and post-interviews.
