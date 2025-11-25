# AIAP Self Practice

## Project Title and Description

*   **Problem Statement:** This project is to help a secondary school identify weaker performing students in the subject of Mathematics so that the school will be able to identify them and help them.
*   **Goal:** Identification of weaker students by the various characteristics that the school is tracking.
*   **Methods Used:** I will perform data ingestion, data cleaning, data standardization, perform feature engineering, Then I will prepare a pipeline to standardize and encode the nominal values. I will use Linear Regression, and from which I will perform levels 1 and 2 regression as part of Feature Selection. With that I will use cross validations to identify right set of parameters before I implement XGBoost.

A dataset was shared with us and the dataset list and attributes are as follows:

| Column               | Description                        |
| -------------------- | ---------------------------------- |
| student_id           | Unique ID for each student         |
| number_of_siblings   | Number of siblings                 |
| direct_admission     | Mode of entering the school        |
| CCA                  | Enrolled CCA                       |
| learning_style       | Primary learning style             |
| tuition              | Indication of whether the student has a tuition   |
| final_test           | Student's O-level mathematics examination score   |
| n_male               | Number of male classmates          |
| n_female             | Number of female classmates        |
| gender               | Gender type                        |
| age                  | Age of the student                 |
| hours_per_week       | Number of hours student studies per week          |
| attendance_rate      | Attendance rate of the student (%) |
| sleep_time           | Daily sleeping time (hour:minutes) |
| wake_time            | Daily waking up time (hour:minutes)               |
| mode_of_transport    | Mode of transport to school        |
| bag_color            | Colours of student's bag           |


## Prerequisites and Installation

The various Python and libraries used in this project are as follows:

| Module/Name | Version |
| ----------- | ------- |
| Python      | 3.11.13 |
| Pandas      | 2.3.1   |
| Numpy       | 2.3.2   |
| Matplotlib  | 3.10.3  |
| Seaborn     | 0.13.2  |
| Conda       | 25.7.0  |
| PyYAML      | 6.0.3   |

There is a custom module called app_logger which I have included in the project. It just needs to be placed at the root directory of the project. When it executes, daily log files will be created in the `log` directory.

````raw
.
├── app_logging
│   └── app_logging.py
├── data
│   └── regression_bonus_practice_data.csv
├── eda.ipynb
├── main.py
├── README.md
└── src
    ├── config.yaml
    ├── data_preparation.py
    └── model_training.py
 ````


## Instructions for Executing the Pipeline

Unpack the files and load it into a directory of your choice. To run the pipeline, several checks need to be performed.

1. Verify Python path

    * Open a Terminal window either on your machine, Follow the steps below.

````raw
(aiap21_tech_asst) admin@admins-MacBook-Pro aiap_foundation_self_practice % which python3
/opt/miniconda3/envs/aiap21_tech_asst/bin/python3
(aiap21_tech_asst) admin@admins-MacBook-Pro aiap_foundation_self_practice % 
````

    * Open the file `main.py` in a text editor. Verify that the first line in the file is the same as the above path. If not, replace the path starting from `#!`.

````python
#!/opt/miniconda3/envs/aiap21_tech_asst/bin/python3

# Import Standard Python Library

# Third-party imports
import pandas as pd
import numpy as np
import yaml
from sklearn.utils._testing import ignore_warnings
from app_logging.app_logging import Logger
````

    * After changing to the right path to the Python executable, don't forget to save it, if necessary.

2. Create a new Conda environment, and making sure that the Python version to be used is **3.11**.

    * From the terminal window, type the following commands to create new environment and then enter the environment.

````raw
(base) admin@admins-MacBook-Pro ~ % conda create -n project python=3.11
2 channel Terms of Service accepted
Retrieving notices: done
Channels:
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done
...
````

    * As Conda is creating the environment, it will pull several modules for starting a new environment.

````raw
...
  tk                 pkgs/main/osx-arm64::tk-8.6.15-hcd8a7d5_0 
  tzdata             pkgs/main/noarch::tzdata-2025b-h04d1e81_0 
  wheel              pkgs/main/osx-arm64::wheel-0.45.1-py311hca03da5_0 
  xz                 pkgs/main/osx-arm64::xz-5.6.4-h80987f9_1 
  zlib               pkgs/main/osx-arm64::zlib-1.3.1-h5f15de7_0 


Proceed ([y]/n)? 
````

    * To proceed, accede to the request and enter `y` followed by **ENTER** key. After pulling and installing the initial required modules, you will be presented with the output below.

````raw
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate project
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(base) admin@admins-MacBook-Pro ~ % 
````

    - To activate the environment, in this case `project`, enter the following command at the terminal window.

````raw
(base) admin@admins-MacBook-Pro ~ % conda activate project
(project) admin@admins-MacBook-Pro ~ % 
````

    * **Note** the change from `base` to `project` 

3. Install required Python modules.

    * To install the required libraries/modules to run this project, locate the directory that you moved or copied the project files to.
    * Ensure that the file `requirements.txt` is in the file.

````raw
(project) admin@admins-MacBook-Pro project % ls -l
total 200
-rw-r--r--@ 1 admin  staff  73853 25 Nov 12:59 eda.ipynb
-rwxr-xr-x@ 1 admin  staff   2925 25 Nov 12:59 main.py
-rw-r--r--@ 1 admin  staff   7891 25 Nov 12:59 README.md
-rw-r--r--@ 1 admin  staff    134 25 Nov 12:59 requirements.txt
-rw-r--r--@ 1 admin  staff   1738 25 Nov 12:59 temp.py
-rw-r--r--@ 1 admin  staff      0 25 Nov 12:59 temp2.ipynb
-rw-r--r--@ 1 admin  staff    573 25 Nov 12:59 Todo.md
(project) admin@admins-MacBook-Pro project % 

````

    * Install the required libraries/modules.

````raw
(project) admin@admins-MacBook-Pro project % pip install -r requirements.txt

...

Installing collected packages: pytz, tzdata, traitlets, threadpoolctl, six, PyYAML, pyparsing, pillow, packaging, numpy, kiwisolver, joblib, fonttools, cycler, scipy, python-dateutil, matplotlib-inline, contourpy, scikit-learn, pandas, matplotlib, seaborn
Successfully installed PyYAML-6.0.2 contourpy-1.3.3 cycler-0.12.1 fonttools-4.60.1 joblib-1.5.2 kiwisolver-1.4.9 matplotlib-3.10.3 matplotlib-inline-0.1.7 numpy-2.3.2 packaging-25.0 pandas-2.3.1 pillow-12.0.0 pyparsing-3.2.5 python-dateutil-2.9.0.post0 pytz-2025.2 scikit-learn-1.7.1 scipy-1.16.1 seaborn-0.13.2 six-1.17.0 threadpoolctl-3.6.0 traitlets-5.14.3 tzdata-2025.2
(project) admin@admins-MacBook-Pro project %
````

4. Execute from the terminal.

   - Using the terminal, navigate to the project directory. 

````raw
abc
````


Provide step-by-step instructions for running your end-to-end machine learning pipeline. Include details on how to execute the 'main.py' script and any other relevant scripts. Additionally, explain how to modify any parameters in the 'config.yaml' file if needed.


## Description of Logical Steps/Flow of the Pipeline

Describe the logical steps and flow of your machine learning pipeline. Include a brief explanation of each major step, such as data cleaning, preprocessing, model training, evaluation, and prediction. If useful, include flow charts or other visual aids to help illustrate the process.


## Overview of Key Findings from EDA

Summarise the key findings from your exploratory data analysis (EDA) conducted in the previous sections. Highlight any important patterns, trends, or insights that influenced the design of your pipeline. Keep the detailed EDA in the Jupyter notebook, and include only a quick summary in the 'README.md'.


## Feature Handling Description

Provide a description of how the features in the dataset are cleaned and processed. Include other relevant information such as any newly engineered features and selected features used in model training. Summarise this information in a table for clarity.


## Explanation of Model Choices

Explain your choice of models for each machine learning task. Discuss why you selected each model and how it fits the specific requirements of your project.


## Evaluation of Models

Evaluate the models you developed, discussing their performance metrics. Explain any metrics used in the evaluation and why they were chosen. This section should provide a clear understanding of how well each model performs.

## Considerations for Deployment

Discuss any additional considerations for deploying the models you developed. This could include scalability, real-time performance, integration with other systems, or any other relevant factors that need to be addressed before deploying the model to production.

