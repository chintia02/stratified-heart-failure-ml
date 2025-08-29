# Heart Failure Events Prediction Stratified by Atrial Fibrillation and Diabetes Status: A Supervised Learning Approach

## Overview

This project focuses on developing predictive models using supervised machine learning,
specifically logistic regression and tree-based algorithms, to predict heart failure events stratified
by atrial fibrillation and diabetes status. The dataset used in this analysis is a large dataset from
the UK primary health care spanning from January 2, 1998, to February 2022 (`data_aurum.csv`).

### Code Structure

This project is organized into two main scripts that cover the entire research workflow, from data analysis to modeling.

* **`Data_analisis_EDA.R`**
    This script is dedicated to **Exploratory Data Analysis (EDA)**. It analyzes data distribution, specifically to understand the prevalence of heart failure events among patients with and without Atrial Fibrillation (AFib) and Diabetes. Key visualizations and descriptive statistics are also generated here.

* **`Modelling_Code_Final.R`**
    This script contains the main code for **machine learning modeling**. Various supervised learning models are trained and evaluated here to predict heart failure events. This script also includes model validation processes and performance analysis.

### Prerequisites

To run the code in this repository, you need to have the following software and R packages installed:

* **R** (version 4.4.2 or later recommended)

The following R packages are also required. You can install them by running this command in your R console:

```R
install.packages(c("dplyr", "tidyr", "caret", "ROCR", "pROC", "PRROC", "randomForest", "ranger", "rpart", "rpart.plot", "DescTools", "reshape2", "car", "tibble", "rms", "xtable", "ggplot2", "ggrepel", "psych", "parallel", "doParallel", "scales"))