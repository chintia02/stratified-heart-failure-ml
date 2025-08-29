#### Load required libraries ####
library(dplyr) #dipake
library(tidyr)
library(caret) #dipake
library(ROCR)
library(pROC) #dipake
library(PRROC) #dipake
library(randomForest)
library(ranger)
library(rpart)
library(rpart.plot)
library(DescTools)
library(reshape2)
library(car)
library(tibble) #dipake
library(rms)
library(xtable)
library(ggplot2)
library(ggrepel)
library(psych)
library(parallel)
library(doParallel)

#### Load the dataset ####
df <- readRDS("df.RDS")
head(df)

# Drop unused column
data <- df

# Dataset-3 consists of 11 features
df3 <- data %>%
  select(
    X_case,
    age_index,
    gender,
    af,
    htn,
    mi,
    depression,
    diabetes,
    valve,
    copd,
    vascular
  )


##### Distinguish Numerical and Categorical Variable for each dataset #####
# Numerical Variable
num_var <- df3 %>% select(c(age_index))
cat_var <- df3 %>% select(!c(age_index))

# Treat the categorical variable as factor
cat_var <- cat_var %>%
  mutate(across(!c(X_case, gender), ~ factor(., levels = c(0, 1))))

cat_var$X_case <- factor(
  cat_var$X_case,
  levels = c("control", "case"),
  labels = c("0", "1")
)

cat_var$gender <- factor(
  cat_var$gender,
  levels = c("M", "F"),
  labels = c("0", "1")
)

##### Combining Categorical and Numerical #####
# Full Dataset
data3 <- cbind(num_var, cat_var)

# AF Subgroup
df_af <- data3 %>% filter(af == 1)
df_non_af <- data3 %>% filter(af == 0)

# Diabetes Subgroup
df_diabetes <- data3 %>% filter(diabetes == 1)
df_non_diabetes <- data3 %>% filter(diabetes == 0)

# Gender Subgroup
df_female <- data3 %>% filter(gender == 1)
df_male <- data3 %>% filter(gender == 0)

#### Ensure outcome as factor and Use the same labels ####
data3$X_case <- factor(
  data3$X_case,
  levels = c(0, 1),
  labels = c("NonHF", "HF")
)

df_af$X_case <- factor(
  df_af$X_case,
  levels = c(0, 1),
  labels = c("NonHF", "HF")
)
df_non_af$X_case <- factor(
  df_non_af$X_case,
  levels = c(0, 1),
  labels = c("NonHF", "HF")
)

df_diabetes$X_case <- factor(
  df_diabetes$X_case,
  levels = c(0, 1),
  labels = c("NonHF", "HF")
)
df_non_diabetes$X_case <- factor(
  df_non_diabetes$X_case,
  levels = c(0, 1),
  labels = c("NonHF", "HF")
)

df_female$X_case <- factor(
  df_female$X_case,
  levels = c(0, 1),
  labels = c("NonHF", "HF")
)
df_male$X_case <- factor(
  df_male$X_case,
  levels = c(0, 1),
  labels = c("NonHF", "HF")
)

##### Contingency Table #####
contingency_table <- list(
  table_af = table(data3$af, data3$X_case),
  table_diabetes = table(data3$diabetes, data3$X_case),
  table_gender = table(data3$gender, data3$X_case)
)

# Print tables
print(contingency_table)

#### Modelling ####
# Step 1: Split Raw Dataset into Train (50%) and Temporary (50%)
train_index <- createDataPartition(data3$X_case, p = 0.5, list = FALSE)
train_data <- data3[train_index, ]
temp_data <- data3[-train_index, ]

# Step 2: Split Temporary set into Validation (50%) and Test (50%)
val_index <- createDataPartition(temp_data$X_case, p = 0.5, list = FALSE)
val_data <- temp_data[val_index, ]
test_data <- temp_data[-val_index, ]

# Check HF, AF, Diabetes, Gender proportion in each subset to ensure fair proportion
# HF Outcomes
prop.table(table(train_data$X_case))
prop.table(table(val_data$X_case))
prop.table(table(test_data$X_case))

# AF
prop.table(table(train_data$af))
prop.table(table(val_data$af))
prop.table(table(test_data$af))

# Diabetes
prop.table(table(train_data$diabetes))
prop.table(table(val_data$diabetes))
prop.table(table(test_data$diabetes))

# Gender
prop.table(table(train_data$gender))
prop.table(table(val_data$gender))
prop.table(table(test_data$gender))

#### Stage 1: Model Selection ####
# Set up Stratified 10-Fold CV for Logit Model
cv_control_logit <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

##### Train Logit Model #####
logit_model <- train(
  X_case ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = cv_control_logit
)

# Set up Stratified 10-Fold CV for Decision Tree and Random Forest
cv_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE
)

##### Parallel Setup for Decision Tree and RF Model Training #####
detectCores()
cluster <- makePSOCKcluster(7)
registerDoParallel(cluster)

##### Train Decision Tree Model #####
system.time(
  dt_model <- train(
    X_case ~ .,
    data = train_data,
    method = "rpart",
    trControl = cv_control
  )
)

##### Train Random Forest Model #####
# Calculate rare class prevalence
rare.class.prevalence <- sum(train_data$X_case == "HF") / nrow(train_data)

# Define class weights based on prevalence for 'ranger' (Inverse weighting)
class_weights <- c(
  "NonHF" = 1 - rare.class.prevalence,
  "HF" = rare.class.prevalence
)

# Train RF using ranger with class weights and balancing
system.time({
  rf_model <- train(
    X_case ~ .,
    data = train_data,
    method = "ranger",
    trControl = cv_control,
    tuneGrid = expand.grid(
      mtry = 3,
      splitrule = "gini",
      min.node.size = 5
    ),
    num.trees = 1000,
    weights = NULL,
    importance = 'impurity',
    class.weights = class_weights
  )
})

# Define sampling sizes per class (based on original example)
sampsize_vec <- c(20000, 4000)
names(sampsize_vec) <- c("NonHF", "HF") # must match factor levels

# Train using randomForest via caret
system.time({
  rf_model2 <- train(
    X_case ~ .,
    data = train_data,
    method = "rf",
    trControl = cv_control,
    tuneGrid = data.frame(mtry = 3),
    ntree = 1000,
    nodesize = 5,
    sampsize = sampsize_vec,
    strata = train_data$X_case,
    cutoff = setNames(c(0.9, 0.1), c("NonHF", "HF"))
  )
})

stopCluster(cluster)

##### Find the Optimal Threshold for Decision Tree using Youden Index #####
dt_roc_obj <- roc(
  val_data$X_case,
  predict(dt_model, newdata = val_data, type = "prob")[, "HF"]
)
dt_coords <- coords(
  dt_roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity")
)
print(dt_coords["threshold"])

#### Evaluate model's Performance ####
evaluate_on_val <- function(model, val_data, threshold = 0.5) {
  # Get predicted probabilities
  probs <- predict(model, newdata = val_data, type = "prob")[, "HF"]

  # Generate predictions based on threshold
  preds <- ifelse(probs > threshold, "HF", "NonHF")

  # Compute confusion matrix
  cm <- confusionMatrix(
    factor(preds, levels = c("NonHF", "HF")),
    val_data$X_case,
    positive = "HF"
  )

  # Compute ROC and PR curve
  roc_obj <- roc(val_data$X_case, probs)
  pr_obj <- pr.curve(
    scores.class0 = probs[val_data$X_case == "HF"],
    scores.class1 = probs[val_data$X_case == "NonHF"],
    curve = FALSE
  )

  # Compute Brier Score
  target_numeric <- as.numeric(val_data$X_case) - 1
  brier_score <- mean((probs - target_numeric)^2)

  # Clamp probabilities away from 0 and 1
  eps <- 1e-15
  probs <- pmin(pmax(probs, eps), 1 - eps)
  # Compute Calibration Slope
  logit_probs <- log(probs / (1 - probs))
  # Fit linear model
  calib_model <- glm(target_numeric ~ logit_probs, family = "binomial")
  calib_slope <- coef(calib_model)["logit_probs"]

  list(
    ConfusionMatrix = cm,
    AUROC = auc(roc_obj),
    PRAUC = pr_obj$auc.integral,
    BrierScore = brier_score,
    CalibrationSlope = calib_slope
  )
}

# Evaluate models
logit_eval <- evaluate_on_val(logit_model, val_data, threshold = 0.00978)
dt_eval <- evaluate_on_val(dt_model, val_data, threshold = 0.01036684)
rf_eval <- evaluate_on_val(rf_model, val_data, threshold = 0.0035)
rf_eval2 <- evaluate_on_val(rf_model2, val_data, threshold = 0.0035)

##### Model's Performance Comparison #####
# Create comparison tibble manually
model_comparison <- bind_rows(
  tibble(
    Model = "Logistic Regression",
    AUROC = as.numeric(logit_eval$AUROC),
    PRAUC = logit_eval$PRAUC,
    Sensitivity = logit_eval$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval$BrierScore,
    CalibrationSlope = logit_eval$CalibrationSlope
  ),
  tibble(
    Model = "Decision Tree",
    AUROC = as.numeric(dt_eval$AUROC),
    PRAUC = dt_eval$PRAUC,
    Sensitivity = dt_eval$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = dt_eval$ConfusionMatrix$byClass["Specificity"],
    BrierScore = dt_eval$BrierScore,
    CalibrationSlope = dt_eval$CalibrationSlope
  ),
  tibble(
    Model = "Random Forest",
    AUROC = as.numeric(rf_eval2$AUROC),
    PRAUC = rf_eval2$PRAUC,
    Sensitivity = rf_eval2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = rf_eval2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = rf_eval2$BrierScore,
    CalibrationSlope = rf_eval2$CalibrationSlope
  )
)

# Show comparison
print(model_comparison)

##### Plot ROC Curve #####
png("AUC-ROC-Final_akhir.png", width = 1200, height = 800)
# Generate and plot ROC curve for logistic regression
roc_obj <- roc(
  val_data$X_case,
  predict(logit_model, newdata = val_data, type = "prob")[, "HF"]
)
plot(
  roc_obj,
  col = "blue",
  lwd = 2,
  main = "ROC Curve Comparison",
  legacy.axes = TRUE
)

# Add ROC curves for other models
lines(
  roc(
    val_data$X_case,
    predict(dt_model, newdata = val_data, type = "prob")[, "HF"]
  ),
  col = "red",
  lwd = 2
)
lines(
  roc(
    val_data$X_case,
    predict(rf_model2, newdata = val_data, type = "prob")[, "HF"]
  ),
  col = "green",
  lwd = 2
)

# Add AUC values to legend
legend(
  "bottomright",
  legend = c(
    paste0("Logistic Regression (AUC = ", round(logit_eval$AUROC, 4), ")"),
    paste0("Decision Tree (AUC = ", round(dt_eval$AUROC, 4), ")"),
    paste0("Random Forest (AUC = ", round(rf_eval2$AUROC, 4), ")")
  ),
  col = c("blue", "red", "green"),
  lwd = 2
)

dev.off()

##### Plot AUC-PR Curve ####
png("PR-AUC-Final.png", width = 1200, height = 800)
# Generate PR curve for logistic regression
probs <- predict(logit_model, newdata = val_data, type = "prob")[, "HF"]
pr_obj <- pr.curve(
  scores.class0 = probs[val_data$X_case == "HF"],
  scores.class1 = probs[val_data$X_case == "NonHF"],
  curve = TRUE
)

plot(pr_obj, col = "blue", lwd = 2, main = "Precision-Recall Curve Comparison")

# Add PR curves for other models
probs_dt <- predict(dt_model, newdata = val_data, type = "prob")[, "HF"]
pr_obj_dt <- pr.curve(
  scores.class0 = probs_dt[val_data$X_case == "HF"],
  scores.class1 = probs_dt[val_data$X_case == "NonHF"],
  curve = TRUE
)
lines(pr_obj_dt$curve[, 1], pr_obj_dt$curve[, 2], col = "red", lwd = 2)

probs_rf <- predict(rf_model2, newdata = val_data, type = "prob")[, "HF"]
pr_obj_rf <- pr.curve(
  scores.class0 = probs_rf[val_data$X_case == "HF"],
  scores.class1 = probs_rf[val_data$X_case == "NonHF"],
  curve = TRUE
)
lines(pr_obj_rf$curve[, 1], pr_obj_rf$curve[, 2], col = "green", lwd = 2)

# Add PR-AUC values to legend
legend(
  "topright",
  legend = c(
    paste0("Logistic Regression (AUC = ", round(logit_eval$PRAUC, 3), ")"),
    paste0("Decision Tree (AUC = ", round(dt_eval$PRAUC, 3), ")"),
    paste0("Random Forest (AUC = ", round(rf_eval2$PRAUC, 3), ")")
  ),
  col = c("blue", "red", "green"),
  lwd = 2
)


dev.off()

#### Stage 2: Model Development ####
##### Combine train and validation set to re-train the best model ####
train_val_data <- rbind(train_data, val_data)

##### Fit the Best Model using train_val data set without stratification #####
##### Fit the Global model using glm() function #####
# Fit the logistic regression model using glm() function
logit_fitted <- glm(
  X_case ~ .,
  data = train_val_data,
  family = binomial()
)

#### Evaluate Fitted model's Performance ####
evaluate_logit_fitted <- function(model, test_data, threshold = 0.5) {
  # Get predicted probabilities
  probs <- predict(model, newdata = test_data, type = "response")

  # Generate predictions based on threshold
  preds <- ifelse(probs > threshold, "HF", "NonHF")

  # Compute confusion matrix
  cm <- confusionMatrix(
    factor(preds, levels = c("NonHF", "HF")),
    test_data$X_case,
    positive = "HF"
  )

  # Compute ROC and PR curve
  roc_obj <- roc(test_data$X_case, probs)
  pr_obj <- pr.curve(
    scores.class0 = probs[test_data$X_case == "HF"],
    scores.class1 = probs[test_data$X_case == "NonHF"],
    curve = FALSE
  )

  # Compute Brier Score
  target_numeric <- as.numeric(test_data$X_case) - 1
  brier_score <- mean((probs - target_numeric)^2)

  # Clamp probabilities away from 0 and 1
  eps <- 1e-15
  probs <- pmin(pmax(probs, eps), 1 - eps)
  # Compute Calibration Slope
  logit_probs <- log(probs / (1 - probs))
  # Fit linear model
  calib_model <- glm(target_numeric ~ logit_probs, family = "binomial")
  calib_slope <- coef(calib_model)["logit_probs"]

  list(
    ConfusionMatrix = cm,
    AUROC = auc(roc_obj),
    PRAUC = pr_obj$auc.integral,
    BrierScore = brier_score,
    CalibrationSlope = calib_slope
  )
}

##### Evaluate Global Model on Subgroup Dataset #####
# AF status
test_data_AF <- subset(test_data, af == 1)
test_data_NonAF <- subset(test_data, af == 0)

# Diabetes status
test_data_Diabetes <- subset(test_data, diabetes == 1)
test_data_NonDiabetes <- subset(test_data, diabetes == 0)

# Gender (Male = 0 / Female = 1)
test_data_Female <- subset(test_data, gender == 1)
test_data_Male <- subset(test_data, gender == 0)

# Evaluate models on AF Subgroup
logit_eval_af <- evaluate_logit_fitted(
  logit_fitted,
  test_data_AF,
  threshold = 0.00978
)
logit_eval_non_af <- evaluate_logit_fitted(
  logit_fitted,
  test_data_NonAF,
  threshold = 0.00978
)

# Evaluate models on Diabetes Subgroup
logit_eval_diabetes <- evaluate_logit_fitted(
  logit_fitted,
  test_data_Diabetes,
  threshold = 0.00978
)
logit_eval_non_diabetes <- evaluate_logit_fitted(
  logit_fitted,
  test_data_NonDiabetes,
  threshold = 0.00978
)

# Evaluate models on Gender Subgroup
logit_eval_female <- evaluate_logit_fitted(
  logit_fitted,
  test_data_Female,
  threshold = 0.00978
)
logit_eval_male <- evaluate_logit_fitted(
  logit_fitted,
  test_data_Male,
  threshold = 0.00978
)

# Create comparison tibble manually
model_comparison_fitted_logit <- bind_rows(
  tibble(
    Model = "Logistic Regression (AF)",
    AUROC = as.numeric(logit_eval_af$AUROC),
    PRAUC = logit_eval_af$PRAUC,
    Sensitivity = logit_eval_af$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_af$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_af$BrierScore,
    CalibrationSlope = logit_eval_af$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Non AF)",
    AUROC = as.numeric(logit_eval_non_af$AUROC),
    PRAUC = logit_eval_non_af$PRAUC,
    Sensitivity = logit_eval_non_af$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_non_af$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_non_af$BrierScore,
    CalibrationSlope = logit_eval_non_af$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Diabetes)",
    AUROC = as.numeric(logit_eval_diabetes$AUROC),
    PRAUC = logit_eval_diabetes$PRAUC,
    Sensitivity = logit_eval_diabetes$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_diabetes$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_diabetes$BrierScore,
    CalibrationSlope = logit_eval_diabetes$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Non Diabetes)",
    AUROC = as.numeric(logit_eval_non_diabetes$AUROC),
    PRAUC = logit_eval_non_diabetes$PRAUC,
    Sensitivity = logit_eval_non_diabetes$ConfusionMatrix$byClass[
      "Sensitivity"
    ],
    Specificity = logit_eval_non_diabetes$ConfusionMatrix$byClass[
      "Specificity"
    ],
    BrierScore = logit_eval_non_diabetes$BrierScore,
    CalibrationSlope = logit_eval_non_diabetes$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Female)",
    AUROC = as.numeric(logit_eval_female$AUROC),
    PRAUC = logit_eval_female$PRAUC,
    Sensitivity = logit_eval_female$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_female$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_female$BrierScore,
    CalibrationSlope = logit_eval_female$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Male)",
    AUROC = as.numeric(logit_eval_male$AUROC),
    PRAUC = logit_eval_male$PRAUC,
    Sensitivity = logit_eval_male$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_male$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_male$BrierScore,
    CalibrationSlope = logit_eval_male$CalibrationSlope
  )
)

# Show comparison
print(model_comparison_fitted_logit)

#writeLines(capture.output(print(model_comparison_fitted_logit)), "model_comparison_fitted_logit.txt")

#### Refit the Best Model using Subgroup data set ####
# AF status
train_val_AF <- subset(train_val_data, af == 1)
train_val_NonAF <- subset(train_val_data, af == 0)

# Diabetes status
train_val_Diabetes <- subset(train_val_data, diabetes == 1)
train_val_NonDiabetes <- subset(train_val_data, diabetes == 0)

# Gender (Male = 0 / Female = 1)
train_val_Female <- subset(train_val_data, gender == 1)
train_val_Male <- subset(train_val_data, gender == 0)


# AF Subgroup
data_af <- train_val_AF %>% select(!c(af))
data_non_af <- train_val_NonAF %>% select(!c(af))

logit_fitted_af <- glm(
  X_case ~ .,
  data = data_af,
  family = binomial()
)

logit_fitted_non_af <- glm(
  X_case ~ .,
  data = data_non_af,
  family = binomial()
)

# Diabetes Subgroup
data_diabetes <- train_val_Diabetes %>% select(!c(diabetes))
data_non_diabetes <- train_val_NonDiabetes %>% select(!c(diabetes))

logit_fitted_diabetes <- glm(
  X_case ~ .,
  data = data_diabetes,
  family = binomial()
)

logit_fitted_non_diabetes <- glm(
  X_case ~ .,
  data = data_non_diabetes,
  family = binomial()
)

# Gender Subgroup
data_female <- train_val_Female %>% select(!c(gender))
data_male <- train_val_Male %>% select(!c(gender))

logit_fitted_female <- glm(
  X_case ~ .,
  data = data_female,
  family = binomial()
)

logit_fitted_male <- glm(
  X_case ~ .,
  data = data_male,
  family = binomial()
)

##### Find the Optimal Threshold for Each Subgrout Dataset using Youden Index #####
# AF
af_roc_obj <- roc(
  test_data_AF$X_case,
  predict(logit_fitted_af, newdata = test_data_AF, type = "response")
)
af_coords <- coords(
  af_roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity")
)
print(af_coords["threshold"])

# Non AF
non_af_roc_obj <- roc(
  test_data_NonAF$X_case,
  predict(logit_fitted_non_af, newdata = test_data_NonAF, type = "response")
)
non_af_coords <- coords(
  non_af_roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity")
)
print(non_af_coords["threshold"])

# Diabetes
diabetes_roc_obj <- roc(
  test_data_Diabetes$X_case,
  predict(
    logit_fitted_diabetes,
    newdata = test_data_Diabetes,
    type = "response"
  )
)
diabetes_coords <- coords(
  diabetes_roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity")
)
print(diabetes_coords["threshold"])

# Non Diabetes
non_diabetes_roc_obj <- roc(
  test_data_NonDiabetes$X_case,
  predict(
    logit_fitted_non_diabetes,
    newdata = test_data_NonDiabetes,
    type = "response"
  )
)
non_diabetes_coords <- coords(
  non_diabetes_roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity")
)
print(non_diabetes_coords["threshold"])

# Female
female_roc_obj <- roc(
  test_data_Female$X_case,
  predict(logit_fitted_female, newdata = test_data_Female, type = "response")
)
female_coords <- coords(
  female_roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity")
)
print(female_coords["threshold"])

# Male
male_roc_obj <- roc(
  test_data_Male$X_case,
  predict(logit_fitted_male, newdata = test_data_Male, type = "response")
)
male_coords <- coords(
  male_roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity")
)
print(male_coords["threshold"])

#### Evaluate Second Fitted Model on Subgroup Dataset ####
# Evaluate models on AF Subgroup
logit_eval_af2 <- evaluate_logit_fitted(
  logit_fitted_af,
  test_data_AF,
  threshold = 0.05588432
)
logit_eval_non_af2 <- evaluate_logit_fitted(
  logit_fitted_non_af,
  test_data_NonAF,
  threshold = 0.00872272
)

# Evaluate models on Diabetes Subgroup
logit_eval_diabetes2 <- evaluate_logit_fitted(
  logit_fitted_diabetes,
  test_data_Diabetes,
  threshold = 0.02566438
)
logit_eval_non_diabetes2 <- evaluate_logit_fitted(
  logit_fitted_non_diabetes,
  test_data_NonDiabetes,
  threshold = 0.008970801
)

# Evaluate models on Gender Subgroup
logit_eval_female2 <- evaluate_logit_fitted(
  logit_fitted_female,
  test_data_Female,
  threshold = 0.01111424
)
logit_eval_male2 <- evaluate_logit_fitted(
  logit_fitted_male,
  test_data_Male,
  threshold = 0.009205854
)

# Create comparison tibble manually
model_comparison_fitted_logit2 <- bind_rows(
  tibble(
    Model = "Logistic Regression (AF)",
    AUROC = as.numeric(logit_eval_af2$AUROC),
    PRAUC = logit_eval_af2$PRAUC,
    Sensitivity = logit_eval_af2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_af2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_af2$BrierScore,
    CalibrationSlope = logit_eval_af2$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Non AF)",
    AUROC = as.numeric(logit_eval_non_af2$AUROC),
    PRAUC = logit_eval_non_af2$PRAUC,
    Sensitivity = logit_eval_non_af2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_non_af2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_non_af2$BrierScore,
    CalibrationSlope = logit_eval_non_af2$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Diabetes)",
    AUROC = as.numeric(logit_eval_diabetes2$AUROC),
    PRAUC = logit_eval_diabetes2$PRAUC,
    Sensitivity = logit_eval_diabetes2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_diabetes2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_diabetes2$BrierScore,
    CalibrationSlope = logit_eval_diabetes2$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Non Diabetes)",
    AUROC = as.numeric(logit_eval_non_diabetes2$AUROC),
    PRAUC = logit_eval_non_diabetes2$PRAUC,
    Sensitivity = logit_eval_non_diabetes2$ConfusionMatrix$byClass[
      "Sensitivity"
    ],
    Specificity = logit_eval_non_diabetes2$ConfusionMatrix$byClass[
      "Specificity"
    ],
    BrierScore = logit_eval_non_diabetes2$BrierScore,
    CalibrationSlope = logit_eval_non_diabetes2$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Female)",
    AUROC = as.numeric(logit_eval_female2$AUROC),
    PRAUC = logit_eval_female2$PRAUC,
    Sensitivity = logit_eval_female2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_female2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_female2$BrierScore,
    CalibrationSlope = logit_eval_female2$CalibrationSlope
  ),
  tibble(
    Model = "Logistic Regression (Male)",
    AUROC = as.numeric(logit_eval_male2$AUROC),
    PRAUC = logit_eval_male2$PRAUC,
    Sensitivity = logit_eval_male2$ConfusionMatrix$byClass["Sensitivity"],
    Specificity = logit_eval_male2$ConfusionMatrix$byClass["Specificity"],
    BrierScore = logit_eval_male2$BrierScore,
    CalibrationSlope = logit_eval_male2$CalibrationSlope
  )
)

# Show comparison
print(model_comparison_fitted_logit2)

#writeLines(capture.output(print(model_comparison_fitted_logit2)), "model_comparison_fitted_logit2.txt")

#### Random forest code from Dr Farag ####
rare.class.prevalence = sum(data3$X_case == 1) / length(data3$X_case) #1 - Balancing by voting rule, AUC of ROC will be unchanged...
#2 - Balancing by sampling stratification
nRareSamples = 1000 * rare.class.prevalence

rf <- randomForest(
  X_case ~ .,
  mtry = 3,
  ntree = 1000,
  nodesize = 5,
  sampsize = c(20000, 4000),
  Strata = data3$X_case,
  data = data3,
  cutoff = setNames(c(0.9, 0.1), c("NonHF", "HF"))
)

evaluate_rf_model <- function(model, test_data, threshold = 0.5) {
  # Get predicted probabilities for class "HF"
  probs <- predict(model, newdata = test_data, type = "prob")[, "HF"]

  # Generate predicted class labels
  preds <- ifelse(probs > threshold, "HF", "NonHF")

  # Confusion Matrix
  cm <- confusionMatrix(
    factor(preds, levels = c("NonHF", "HF")),
    test_data$X_case,
    positive = "HF"
  )

  # ROC and PR curve
  roc_obj <- roc(test_data$X_case, probs)
  pr_obj <- pr.curve(
    scores.class0 = probs[test_data$X_case == "HF"],
    scores.class1 = probs[test_data$X_case == "NonHF"],
    curve = FALSE
  )

  # Brier Score
  target_numeric <- as.numeric(test_data$X_case) - 1
  brier_score <- mean((probs - target_numeric)^2)

  # Calibration Slope
  eps <- 1e-15
  probs_clamped <- pmin(pmax(probs, eps), 1 - eps)
  logit_probs <- log(probs_clamped / (1 - probs_clamped))
  calib_model <- glm(target_numeric ~ logit_probs, family = "binomial")
  calib_slope <- coef(calib_model)["logit_probs"]

  list(
    ConfusionMatrix = cm,
    AUROC = auc(roc_obj),
    PRAUC = pr_obj$auc.integral,
    BrierScore = brier_score,
    CalibrationSlope = calib_slope
  )
}

rf_eval <- evaluate_rf_model(rf, test_data = data3, threshold = 0.0035)

rf_performance <- tibble(
  Model = "Random Forest",
  AUROC = as.numeric(rf_eval$AUROC),
  PRAUC = rf_eval$PRAUC,
  Sensitivity = rf_eval$ConfusionMatrix$byClass["Sensitivity"],
  Specificity = rf_eval$ConfusionMatrix$byClass["Specificity"],
  BrierScore = rf_eval$BrierScore,
  CalibrationSlope = rf_eval$CalibrationSlope
)

print(rf_performance)
