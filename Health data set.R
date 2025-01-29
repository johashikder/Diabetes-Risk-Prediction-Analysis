# Load Required Libraries
install.packages(c("tidyverse", "caret", "randomForest", "corrplot", 
                   "pROC", "GGally", "gridExtra"))
library(tidyverse)
library(caret)
library(randomForest)
library(corrplot)
library(pROC)
library(GGally)
library(gridExtra)

# Load Dataset
df <- read.csv("health_dataset.csv", stringsAsFactors = TRUE)

# Check Data Structure
str(df)
summary(df)

# Handling Missing Values
df <- na.omit(df)

# Convert Categorical Variables to Factors
categorical_vars <- c("Gender", "Ethnicity", "SocioeconomicStatus","EducationLevel", 
                      "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality", 
                      "SleepQuality", "FamilyHistoryDiabetes", "GestationalDiabetes",
                      "PolycysticOvarySyndrome", "PreviousPreDiabetes", "Hypertension",
                      "AntihypertensiveMedications", "Statins", "AntidiabeticMedications",
                      "FrequentUrination", "ExcessiveThirst", "UnexplainedWeightLoss", 
                      "BlurredVision", "SlowHealingSores", "TinglingHandsFeet", "Diagnosis")

df[categorical_vars] <- lapply(df[categorical_vars], as.factor)

# Exploratory Data Analysis (EDA)
## Histogram of Age
ggplot(df, aes(x = Age, fill = Diagnosis)) + 
  geom_histogram(bins = 20, alpha = 0.7) + 
  theme_minimal() + 
  ggtitle("Age Distribution by Diabetes Diagnosis")

## Boxplot of BMI vs. Diabetes Diagnosis
ggplot(df, aes(x = Diagnosis, y = BMI, fill = Diagnosis)) + 
  geom_boxplot() + 
  theme_minimal() + 
  ggtitle("BMI vs Diabetes Diagnosis")

# Correlation Analysis (Numerical Variables)
numeric_vars <- df %>% select_if(is.numeric)
corr_matrix <- cor(numeric_vars, use = "complete.obs")
corrplot(corr_matrix, method = "color", tl.cex = 0.7)

# Splitting Data into Train and Test (80% Train, 20% Test)
set.seed(123)
trainIndex <- createDataPartition(df$Diagnosis, p = 0.8, list = FALSE)
train_data <- df[trainIndex, ]
test_data <- df[-trainIndex, ]

# Train Random Forest Model for Diabetes Prediction
set.seed(123)
rf_model <- randomForest(Diagnosis ~ ., data = train_data, ntree = 100, importance = TRUE)

# Model Summary
print(rf_model)

# Feature Importance
importance_df <- data.frame(Feature = rownames(importance(rf_model)), Importance = importance(rf_model)[, 1])
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat = "identity") + 
  coord_flip() + 
  theme_minimal() + 
  ggtitle("Feature Importance for Diabetes Prediction")

# Predictions on Test Data
predictions <- predict(rf_model, test_data)

# Model Performance Metrics
conf_matrix <- confusionMatrix(predictions, test_data$Diagnosis)
print(conf_matrix)

# ROC Curve
roc_curve <- roc(test_data$Diagnosis, as.numeric(predictions))
plot(roc_curve, col = "blue", main = "ROC Curve for Diabetes Prediction")

# Save Model
saveRDS(rf_model, "diabetes_prediction_model.rds")

# Export Predictions
write.csv(data.frame(Actual = test_data$Diagnosis, Predicted = predictions), "predictions.csv", row.names = FALSE)

