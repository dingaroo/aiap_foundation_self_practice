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

There is a custom module called app_logger which I have included in the project. It just needs to be placed at the root directory of the project. When it executes, daily log files will be created in the `log` directory.

```raw
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
 ```

List any prerequisites needed to run the project, such as Python version, libraries, and other dependencies. Include instructions on how to set up the environment and install the necessary packages. This is where your 'requirements.txt' file will come in handy.




## Instructions for Executing the Pipeline

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

