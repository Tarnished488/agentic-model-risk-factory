# GMSC LightGBM: R reference export
# Original R code: 'Top 1 - Give Me Some Credit: LGBM model in R' by brynning, v1 (Kaggle, Apache 2.0), copied unchanged except
# for data/output paths. The final block is added to export parity references.
# Run from the repository root:  Rscript scripts/<this file>
R.version.string
# Rscript cannot create a personal library on its own; use it so installs need no admin rights.
dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
pkgs <- c("dplyr", "lightgbm")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
pdf(NULL)  # discard plots when run with Rscript
data_dir <- Sys.getenv("GMSC_DIR", "data/GiveMeSomeCredit")
out <- Sys.getenv("OUT_DIR", "data/reference/gmsc_lightgbm")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# ---- cell 1 ----
## Load packages
library(dplyr)
library(lightgbm)

## Load Training and Test data
TrainingData <- read.csv(file.path(data_dir, "cs-training.csv"))
TestData <- read.csv(file.path(data_dir, "cs-test.csv"))

## Remove coloumn X
TrainingData <- TrainingData %>% select(-X)

## Outlier
TrainingData$age[which(TrainingData$age == 0)] <- round(mean(TrainingData$age[TrainingData$age != 0]))

## Manual Variables
TrainingData <- TrainingData %>% mutate(TotalPastDue = NumberOfTime30.59DaysPastDueNotWorse + 
                                          NumberOfTime60.89DaysPastDueNotWorse + NumberOfTimes90DaysLate,
                                        DebtRatio1000= ifelse(MonthlyIncome < 1000, NA, DebtRatio))

## Function for general transformations
feature_engineering <- function(data, variables, operations = c("subtract")) {
  # data: The input data frame
  # variables: A vector of variable names for feature engineering
  # operations: A vector of operations to perform ("subtract", "divide", "log", "square", ...)
  
  for (var in variables) {
    if (var %in% colnames(data)) {
      for (operation in operations) {
        if (operation == "subtract") {
          new_var <- paste0(var, "_minus_mean")
          data[[new_var]] <- data[[var]] - mean(data[[var]], na.rm = TRUE)
        } else if (operation == "divide" && mean(data[[var]], na.rm = TRUE) != 0) {
          new_var <- paste0(var, "_divided_by_mean")
          data[[new_var]] <- data[[var]] / mean(data[[var]], na.rm = TRUE)
        } else if (operation == "log") {
          new_var <- paste0("log_", var)
          data[[new_var]] <- ifelse(data[[var]] == 0, NA, log(data[[var]]))
        } else if (operation == "square") {
          new_var <- paste0(var, "_squared")
          data[[new_var]] <- data[[var]]^2
        } else {
          warning("Unsupported operation '", operation, "'. Skipping.")
        }
      }
    } else {
      warning("Variable '", var, "' not found in the data.")
    }
  }
  
  return(data)
}

## Function for combining 2 variables
arithmetic_operations <- function(data, num_var1, num_var2, operations = c("divide")) {
  # data: The input data frame
  # num_var1, num_var2: Names of the numeric variables for arithmetic operation
  # operations: A vector of arithmetic operations to perform ("multiply", "subtract", "divide", "add")
  
  new_vars <- character(length(operations))
  
  if (num_var1 %in% colnames(data) && num_var2 %in% colnames(data)) {
    for (i in seq_along(operations)) {
      operation <- operations[i]
      
      new_vars[i] <- switch(
        operation,
        multiply = paste0(num_var1, "_times_", num_var2),
        subtract = paste0(num_var1, "_minus_", num_var2),
        divide = paste0(num_var1, "_divided_by_", num_var2),
        add = paste0(num_var1, "_plus_", num_var2),
        warning("Unsupported operation '", operation, "'.")
      )
      
      if (operation == "multiply") {
        data[[new_vars[i]]] <- data[[num_var1]] * data[[num_var2]]
      } else if (operation == "subtract") {
        data[[new_vars[i]]] <- data[[num_var1]] - data[[num_var2]]
      } else if (operation == "divide") {
        data[[new_vars[i]]] <- ifelse(is.na(data[[num_var2]]) | data[[num_var2]] == 0, NA, data[[num_var1]] / data[[num_var2]])
      } else if (operation == "add") {
        data[[new_vars[i]]] <- data[[num_var1]] + data[[num_var2]]
      } else {
        warning("Error during arithmetic operation.")
      }
    }
  } else {
    warning("One or both of the variables not found in the data.")
  }
  
  return(cbind(data))
}

## Select variables to transform
variables_to_engineer <- colnames(TrainingData %>% select(-c(SeriousDlqin2yrs)))

## Transform variables using subtract, divide, log and square.
TrainingData <- feature_engineering(TrainingData, variables_to_engineer, operation = c("subtract","divide","log","square"))

## Generate which variables to combine (We dont use the newly generated variables from previous step)
variable_combinations <- combn(variables_to_engineer, 2, simplify = TRUE)

## Loop over all combinations
for (i in seq_len(ncol(variable_combinations))) {
  var1 <- variable_combinations[1, i]
  var2 <- variable_combinations[2, i]
  TrainingData <- arithmetic_operations(TrainingData, var1, var2, operations = c("multiply", "subtract", "divide", "add"))
}


####### Variables for TestData
TestData <- TestData %>% mutate(TotalPastDue = NumberOfTime30.59DaysPastDueNotWorse + 
                                  NumberOfTime60.89DaysPastDueNotWorse + NumberOfTimes90DaysLate,
                                DebtRatio1000= ifelse(MonthlyIncome < 1000, NA, DebtRatio))


variables_to_engineer <- colnames(TestData %>% select(-c(X,SeriousDlqin2yrs)))

TestData <- feature_engineering(TestData, variables_to_engineer, operation = c("subtract","divide","log","square"))

variable_combinations <- combn(variables_to_engineer, 2, simplify = TRUE)

for (i in seq_len(ncol(variable_combinations))) {
  var1 <- variable_combinations[1, i]
  var2 <- variable_combinations[2, i]
  
  TestData <- arithmetic_operations(TestData, var1, var2, operations = c("multiply", "subtract", "divide", "add"))
}

# ---- cell 2 ----
# Create a sparse matrix without removing NA rows as LGBM handles NA 
sparse_matrix <- model.matrix.lm(SeriousDlqin2yrs ~ ., TrainingData, na.action = "na.pass")

# Create the training data list
train <- list(data = sparse_matrix, label = as.numeric(TrainingData$SeriousDlqin2yrs))


# Set parameters for model training
train_params <- list(
  num_leaves = 12L
  , learning_rate = 0.013
  , objective = "binary"
  , feature_fraction = c(0.8)
  , use_missing = TRUE
  , metric = "AUC"
  , lambda_l1 = 0.1
  , lambda_l2 = 10
)

## LGBM model
bst <- lightgbm(
  data = train$data
  , params = train_params
  , label = train$label
  , nrounds = 750L
)

# ---- cell 3 ----

## Feature importance
feature_importance <- lgb.importance(model = bst)

## Select top 100 features from LGBM model
selected_columns <- c("SeriousDlqin2yrs", feature_importance$Feature[1:100])

## Create new Training and Test data using only these features
TrainingData <- TrainingData[, selected_columns, drop = FALSE]
TestData <- TestData[, c("X",selected_columns), drop = FALSE]

# Create a sparse matrix without removing NA rows
sparse_matrix <- model.matrix.lm(SeriousDlqin2yrs ~ ., TrainingData, na.action = "na.pass")

# Create the training data list
train <- list(data = sparse_matrix, label = as.numeric(TrainingData$SeriousDlqin2yrs))

# Set parameters for model training
train_params <- list(
  num_leaves = 12L
  , learning_rate = 0.0135
  , objective = "binary"
  , feature_fraction = c(0.7)
  , use_missing = TRUE
  , metric = "AUC"
  , lambda_l1 = 0.1
  , lambda_l2 = 4
  , bagging_freq = 10
  , bagging_fraction = 0.8
)

#--------------------Basic Training using lightgbm----------------
print("Training lightgbm with sparseMatrix")
bst <- lightgbm(
  data = train$data
  , params = train_params
  , label = train$label
  , nrounds = 1000L
)

## Prepare the testdata
TestData2 <- TestData %>% select(-c(X))

# Create a sparse matrix without removing NA rows
sparse_matrix <- model.matrix.lm(SeriousDlqin2yrs ~ ., TestData2, na.action = "na.pass")

# Create the test data list
test <- list(
  data = sparse_matrix,
  label = as.numeric(TestData2$SeriousDlqin2yrs)
)

#--------------------Basic prediction using lightgbm--------------
# You can do prediction using the following line
# You can put in Matrix, sparseMatrix, or lgb.Dataset
pred <- predict(bst, test$data)


Result <- data.frame(Id = as.integer(TestData$X), Probability = pred)

write.csv(Result,file.path(out, "LGBM_Summit.csv"), row.names = FALSE)


# ---- added: reference export ----
lgb.save(bst, file.path(out, "model.txt"))
write.csv(data.frame(row_id = as.integer(TestData$X), probability = pred),
          file.path(out, "test_predictions.csv"), row.names = FALSE)
writeLines(colnames(test$data), file.path(out, "feature_order.txt"))
write.csv(data.frame(row_id = as.integer(TestData$X[1:2000]), as.matrix(test$data[1:2000, ]), check.names = FALSE),
          file.path(out, "test_features_head.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
print(list.files(out))
