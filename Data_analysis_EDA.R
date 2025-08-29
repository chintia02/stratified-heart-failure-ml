# Set working directory
setwd("N:/uolstore/Research/b/b0168/Data/proj3/Data Analysis")

#### Load required libraries ####
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(psych)
library(ggrepel)


# load the dataset
#df <- read.csv("data_aurum.csv")
#saveRDS(df, file="df.RDS")

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

#### Descriptive and Visual Exploration ####
# Check the data structure
str(df3)

# Check missing value
colSums(is.na(df3))
which(colSums(is.na(df3)) > 0)

# Write the output as txt file
missing_value <- list(
  "Missing Columns" = which(colSums(is.na(df3)) > 0),
  "NA Counts per Column" = colSums(is.na(df3))
)

#writeLines(capture.output(print(missing_value)), "missing_value_df3.txt")

##### Distinguish Numerical and Categorical Variable for each dataset#####
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

# Add Variable Age Group: Classify age into three clinically meaningful categories
data3 <- data3 %>%
  mutate(
    age_group = case_when(
      age_index < 65 ~ "<65",
      age_index >= 65 & age_index <= 75 ~ "65–75",
      age_index > 75 ~ ">75"
    )
  )

##### Contingency Table #####
contingency_table <- list(
  table_gender = table(cat_var$gender, cat_var$X_case),
  table_af = table(cat_var$af, cat_var$X_case),
  table_htn = table(cat_var$htn, cat_var$X_case),
  table_mi = table(cat_var$mi, cat_var$X_case),
  table_depression = table(cat_var$depression, cat_var$X_case),
  table_diabetes = table(cat_var$diabetes, cat_var$X_case),
  table_valve = table(cat_var$valve, cat_var$X_case),
  table_copd = table(cat_var$copd, cat_var$X_case),
  table_vascular = table(cat_var$vascular, cat_var$X_case)
)

# Print tables
print(contingency_table)

##### Chi-Squared Test #####
chisq_test <- list(
  chisq_test_gender = chisq.test(contingency_table$table_gender),
  chisq_test_af = chisq.test(contingency_table$table_af),
  chisq_test_htn = chisq.test(contingency_table$table_htn),
  chisq_test_mi = chisq.test(contingency_table$table_mi),
  chisq_test_depression = chisq.test(contingency_table$table_depression),
  chisq_test_diabetes = chisq.test(contingency_table$table_diabetes),
  chisq_test_valve = chisq.test(contingency_table$table_valve),
  chisq_test_copd = chisq.test(contingency_table$table_copd),
  chisq_test_vascular = chisq.test(contingency_table$table_vascular)
)

# View results
print(chisq_test)

#### Adjusted Odds Ratio ####
##### Univariate #####
# List of predictors
predictors <- c(
  "age_index",
  "gender",
  "af",
  "htn",
  "mi",
  "depression",
  "diabetes",
  "valve",
  "copd",
  "vascular"
)

# Initialize empty list to store results
univariate_results <- list()

# Loop through each predictor
for (var in predictors) {
  # Fit univariate logistic regression
  formula <- as.formula(paste("X_case ~", var))
  model <- glm(formula, data = data3, family = binomial)

  # Extract coefficient summary
  coef_summary <- summary(model)$coefficients

  # Extract values for the predictor (row 2)
  logit_coef <- coef_summary[2, "Estimate"]
  std_error <- coef_summary[2, "Std. Error"]
  p_value <- coef_summary[2, "Pr(>|z|)"]

  # Odds ratio and confidence interval
  OR <- exp(logit_coef)
  CI <- exp(confint(model)[2, ]) # CI for the predictor

  # Store results
  univariate_results[[var]] <- data.frame(
    Variable = var,
    Odds_Ratio = round(OR, 4),
    CI_Lower = round(CI[1], 4),
    CI_Upper = round(CI[2], 4),
    Std_Error = round(std_error, 4),
    P_Value = signif(p_value, 3)
  )
}

# Combine all results into one data frame
univariate_summary <- do.call(rbind, univariate_results)

# Print the summary table
print(univariate_summary)


##### Multivariate #####
# Fit the logistic regression model
logit_model <- glm(
  X_case ~
    age_index +
      gender +
      af +
      htn +
      mi +
      depression +
      diabetes +
      valve +
      copd +
      vascular,
  data = data3,
  family = binomial
)

# Logit coefficients
logit_coef <- coef(logit_model)

# Odds ratios
odds_ratios <- exp(logit_coef)

# Confidence intervals on logit scale
conf_int <- confint(logit_model)

# Convert to odds ratio scale
odds_ratio_ci <- exp(conf_int)

# Create summary table
summary_table <- data.frame(
  Variable = names(odds_ratios),
  Odds_Ratio = round(odds_ratios, 4),
  CI_Lower = round(odds_ratio_ci[, 1], 4),
  CI_Upper = round(odds_ratio_ci[, 2], 4)
)

print(summary_table)

#writeLines(capture.output(print(summary_table)), "odds_ratio_summary_table.txt") # nolint

##### Visualisation for numerical variable #####
summary(num_var)

# Summary of Age by Group
summary_age_by_group <- describeBy(data3$age_index, group = data3$X_case)
print(summary_age_by_group)

# writeLines(capture.output(print(summary(num_var))), "age_summary.txt") # nolint

# Density Distribution
num_var_names <- names(num_var)

# Convert data to long format for ggplot
long_data <- data3 %>%
  pivot_longer(
    cols = all_of(num_var_names),
    names_to = "variable",
    values_to = "value"
  )

#png("age_histogram_by_case_new.png", width = 1200, height = 800)
# Create histogram and density plots for each numerical variable
ggplot(long_data, aes(x = value, fill = factor(X_case))) +
  geom_histogram(
    aes(y = ..density..),
    position = "identity",
    alpha = 0.3,
    bins = 30,
    color = "white"
  ) +
  geom_density(alpha = 0.5) +
  facet_wrap(~variable, scales = "free") +
  geom_vline(
    data = long_data %>%
      group_by(variable, X_case) %>%
      summarise(
        mean_val = mean(value),
        median_val = median(value),
        .groups = "drop"
      ),
    aes(xintercept = mean_val, color = factor(X_case)),
    linetype = "dashed",
    linewidth = 1
  ) +
  geom_vline(
    data = long_data %>%
      group_by(variable, X_case) %>%
      summarise(
        mean_val = mean(value),
        median_val = median(value),
        .groups = "drop"
      ),
    aes(xintercept = median_val, color = factor(X_case)),
    linetype = "solid",
    linewidth = 1
  ) +
  labs(
    title = "Distribution of Age by Heart Failure Group",
    x = "Value",
    y = "Density",
    fill = "Outcome (HF Case)",
    color = "Outcome (HF Case)"
  ) +
  theme_minimal()

#dev.off()

# Data Visualisation: Histogram for continuous variables

#png("age_histogram.png", width = 1200, height = 800)

# Create Histogram
hist(
  df[["age_index"]],
  main = paste("Histogram of Patients' Age"),
  xlab = "age_index",
  col = "skyblue",
  border = "black",
  breaks = 30,
  probability = TRUE
)

# Add density curve
lines(density(df[["age_index"]], na.rm = TRUE), col = "darkblue", lwd = 2)

# Add vertical lines for mean and median
abline(
  v = mean(df[["age_index"]], na.rm = TRUE),
  col = "darkgreen",
  lwd = 2,
  lty = 2
)
abline(
  v = median(df[["age_index"]], na.rm = TRUE),
  col = "red",
  lwd = 2,
  lty = 3
)

#dev.off()

# Data Visualisation: Boxlplot for continuous variable

#png("age_boxplot_by_case.png", width = 1200, height = 800)

# Create Boxplot of Age by HF Outcome
ggplot(data1, aes(x = X_case, y = age_index, fill = X_case)) +
  geom_boxplot() +
  labs(
    title = "Distribution of Age by HF Outcome",
    x = "HF Event",
    y = "Age"
  ) +
  theme_minimal()

#dev.off()

##### Visualisation for categorical variables #####
# Calculate the count, percentage, and y position
plot_data1 <- data3 %>%
  gather(key = "variable", value = "value", c(gender, af, htn)) %>%
  group_by(variable, value, X_case) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(variable, value) %>%
  mutate(
    percentage = round(100 * count / sum(count), 1)
  ) %>%
  arrange(X_case, .by_group = TRUE) %>%
  mutate(
    ypos = ifelse(
      X_case == "0", # segmen bawah di tengah batang
      count / 2,
      sum(count) * 1 # segmen atas sedikit di atas total tinggi batang
    )
  ) %>%
  ungroup()

# Define custom colors
fill_colors <- c("0" = "#a6cee3", "1" = "#1f78b4")
text_colors <- c("0" = "#1f78b4", "1" = "#08306b")

plot_data1$X_case <- factor(plot_data1$X_case, levels = c("1", "0"))

#png("categorical_dist1_new_final.png", width = 1200, height = 800)

# Plot
ggplot(
  plot_data1,
  aes(x = factor(value, levels = c(1, 0)), y = count, fill = X_case)
) +
  geom_bar(stat = "identity", position = "stack") +

  # Label otomatis berdasarkan ypos
  geom_text_repel(
    aes(
      y = ypos,
      label = paste0(count, " (", percentage, "%)"),
      color = X_case
    ),
    size = 3,
    show.legend = FALSE
  ) +

  facet_wrap(~variable, scales = "free") +
  scale_fill_manual(values = fill_colors) +
  scale_color_manual(values = text_colors) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.1)),
    labels = label_number(accuracy = 1, big.mark = "") # angka full tanpa koma
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal() +
  labs(
    x = "Value for af and htn (0 = No, 1 = Yes); Value for gender (0 = Male, 1 = Female)",
    y = "Count",
    fill = "Outcome (HF Case)",
    title = "Figure 1(a): Categorical Variable Distribution by Outcome"
  )

#dev.off()

plot_data2 <- data3 %>%
  gather(key = "variable", value = "value", c(mi, depression, diabetes)) %>%
  group_by(variable, value, X_case) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(variable, value) %>%
  mutate(
    percentage = round(100 * count / sum(count), 1)
  ) %>%
  arrange(X_case, .by_group = TRUE) %>%
  mutate(
    ypos = ifelse(
      X_case == "0", # segmen bawah di tengah batang
      count / 2,
      sum(count) * 1 # segmen atas sedikit di atas total tinggi batang
    )
  ) %>%
  ungroup()

plot_data2$X_case <- factor(plot_data2$X_case, levels = c("1", "0"))

#png("categorical_dist2_new_final.png", width = 1200, height = 800)

# Plot
ggplot(
  plot_data2,
  aes(x = factor(value, levels = c(1, 0)), y = count, fill = X_case)
) +
  geom_bar(stat = "identity", position = "stack") +

  # Label otomatis berdasarkan ypos
  geom_text_repel(
    aes(
      y = ypos,
      label = paste0(count, " (", percentage, "%)"),
      color = X_case
    ),
    size = 3,
    show.legend = FALSE
  ) +

  facet_wrap(~variable, scales = "free") +
  scale_fill_manual(values = fill_colors) +
  scale_color_manual(values = text_colors) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.1)),
    labels = label_number(accuracy = 1, big.mark = "") # angka full tanpa koma
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal() +
  labs(
    x = "Value (0 = No, 1 = Yes)",
    y = "Count",
    fill = "Outcome (HF Case)",
    title = "Figure 1(b): Categorical Variable Distribution by Outcome"
  )

#dev.off()

plot_data3 <- data3 %>%
  gather(key = "variable", value = "value", c(valve, copd, vascular)) %>%
  group_by(variable, value, X_case) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(variable, value) %>%
  mutate(
    percentage = round(100 * count / sum(count), 1)
  ) %>%
  arrange(X_case, .by_group = TRUE) %>%
  mutate(
    ypos = ifelse(
      X_case == "0", # segmen bawah di tengah batang
      count / 2,
      sum(count) * 1 # segmen atas sedikit di atas total tinggi batang
    )
  ) %>%
  ungroup()

plot_data3$X_case <- factor(plot_data3$X_case, levels = c("1", "0"))

#png("categorical_dist3_new_final.png", width = 1200, height = 800)

# Plot
ggplot(
  plot_data3,
  aes(x = factor(value, levels = c(1, 0)), y = count, fill = X_case)
) +
  geom_bar(stat = "identity", position = "stack") +

  # Label otomatis berdasarkan ypos
  geom_text_repel(
    aes(
      y = ypos,
      label = paste0(count, " (", percentage, "%)"),
      color = X_case
    ),
    size = 3,
    show.legend = FALSE,
    direction = "both",
    nudge_y = 1
  ) +

  facet_wrap(~variable, scales = "free") +
  scale_fill_manual(values = fill_colors) +
  scale_color_manual(values = text_colors) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.1)),
    labels = label_number(accuracy = 1, big.mark = "") # angka full tanpa koma
  ) +
  coord_cartesian(clip = "off") +
  theme_minimal() +
  labs(
    x = "Value (0 = No, 1 = Yes)",
    y = "Count",
    fill = "Outcome (HF Case)",
    title = "Figure 1(c): Categorical Variable Distribution by Outcome"
  )

#dev.off()

##### Interaction Analysis #####
gender_interaction <- list(
  gender_drug_antihtn = glm(
    X_case ~ gender * drug_antihtn,
    data = cat_var,
    family = binomial
  ),
  gender_drug_diabetes = glm(
    X_case ~ gender * drug_diabetes,
    data = cat_var,
    family = binomial
  ),
  gender_smoking_current = glm(
    X_case ~ gender * smoking_current,
    data = cat_var,
    family = binomial
  ),
  gender_smoking_ex = glm(
    X_case ~ gender * smoking_ex,
    data = cat_var,
    family = binomial
  ),
  gender_af = glm(X_case ~ gender * af, data = cat_var, family = binomial),
  gender_angina = glm(
    X_case ~ gender * angina,
    data = cat_var,
    family = binomial
  ),
  gender_htn = glm(X_case ~ gender * htn, data = cat_var, family = binomial),
  gender_anaemia = glm(
    X_case ~ gender * anaemia,
    data = cat_var,
    family = binomial
  ),
  gender_stroke = glm(
    X_case ~ gender * stroke,
    data = cat_var,
    family = binomial
  ),
  gender_cancer = glm(
    X_case ~ gender * cancer,
    data = cat_var,
    family = binomial
  ),
  gender_mi = glm(X_case ~ gender * mi, data = cat_var, family = binomial),
  gender_depression = glm(
    X_case ~ gender * depression,
    data = cat_var,
    family = binomial
  ),
  gender_autoimmune = glm(
    X_case ~ gender * autoimmune,
    data = cat_var,
    family = binomial
  ),
  gender_pvd = glm(X_case ~ gender * pvd, data = cat_var, family = binomial),
  gender_diabetes = glm(
    X_case ~ gender * diabetes,
    data = cat_var,
    family = binomial
  ),
  gender_valve = glm(
    X_case ~ gender * valve,
    data = cat_var,
    family = binomial
  ),
  gender_copd = glm(X_case ~ gender * copd, data = cat_var, family = binomial),
  gender_vascular = glm(
    X_case ~ gender * vascular,
    data = cat_var,
    family = binomial
  )
)

age_interaction <- list(
  age_drug_antihtn = glm(
    X_case ~ age_group * drug_antihtn,
    data = data1,
    family = binomial
  ),
  age_drug_diabetes = glm(
    X_case ~ age_group * drug_diabetes,
    data = data1,
    family = binomial
  ),
  age_smoking_current = glm(
    X_case ~ age_group * smoking_current,
    data = data1,
    family = binomial
  ),
  age_smoking_ex = glm(
    X_case ~ age_group * smoking_ex,
    data = data1,
    family = binomial
  ),
  age_af = glm(X_case ~ age_group * af, data = data1, family = binomial),
  age_angina = glm(
    X_case ~ age_group * angina,
    data = data1,
    family = binomial
  ),
  age_htn = glm(X_case ~ age_group * htn, data = data1, family = binomial),
  age_anaemia = glm(
    X_case ~ age_group * anaemia,
    data = data1,
    family = binomial
  ),
  age_stroke = glm(
    X_case ~ age_group * stroke,
    data = data1,
    family = binomial
  ),
  age_cancer = glm(
    X_case ~ age_group * cancer,
    data = data1,
    family = binomial
  ),
  age_mi = glm(X_case ~ age_group * mi, data = data1, family = binomial),
  age_depression = glm(
    X_case ~ age_group * depression,
    data = data1,
    family = binomial
  ),
  age_autoimmune = glm(
    X_case ~ age_group * autoimmune,
    data = data1,
    family = binomial
  ),
  age_pvd = glm(X_case ~ age_group * pvd, data = data1, family = binomial),
  age_diabetes = glm(
    X_case ~ age_group * diabetes,
    data = data1,
    family = binomial
  ),
  age_valve = glm(X_case ~ age_group * valve, data = data1, family = binomial),
  age_copd = glm(X_case ~ age_group * copd, data = data1, family = binomial),
  age_vascular = glm(
    X_case ~ age_group * vascular,
    data = data1,
    family = binomial
  )
)

# Print summary for each model in the interaction list
gender_interaction_summary <- lapply(gender_interaction, summary)
age_interaction_summary <- lapply(age_interactiont, summary)

print(gender_interaction_summary)
print(age_interaction_summary)

# writeLines(capture.output(print(gender_interaction_summary)), "gender_interaction.txt")
# writeLines(capture.output(print(age_interaction_summary)), "age_interaction.txt")

htn_diabetes_interaction <- glm(
  X_case ~ htn * diabetes,
  data = data1,
  family = binomial
)
summary(htn_diabetes_interaction)

#writeLines(capture.output(summary(htn_diabetes_interaction)), "htn_diabetes_interaction.txt")
