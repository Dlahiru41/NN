# Load required libraries
library(neuralnet)
library(tidyverse)
library(lubridate)
library(Metrics)

# Read the data
data <- read_excel("C://Users//dlahi//OneDrive//Desktop//Assigments//ML ref def//ExchangeUSD.xlsx")

# Ensure the date column is in date format
data$Date <- as.Date(data$`YYYY/MM/DD`)

# Select only the USD/EUR column and create a time series
ts_data <- ts(data$`USD/EUR`, frequency = 365)

# Split the data into training (400) and testing (100) sets
train_data <- ts_data[1:400]
test_data <- ts_data[401:500]

# Function to create input/output matrices
create_io_matrix <- function(data, lag_order) {
  n <- length(data)
  x <- matrix(nrow = n - lag_order, ncol = lag_order)
  y <- vector(length = n - lag_order)

  for (i in 1:(n - lag_order)) {
    x[i, ] <- data[i:(i + lag_order - 1)]
    y[i] <- data[i + lag_order]
  }

  colnames(x) <- paste0("lag_", 1:lag_order)

  return(list(x = as.data.frame(x), y = y))
}

# Create I/O matrices for different lag orders
lag_orders <- c(1, 5, 7, 10)
io_matrices <- lapply(lag_orders, function(lag) create_io_matrix(train_data, lag))

# Normalize function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Normalize I/O matrices
normalized_matrices <- lapply(io_matrices, function(mat) {
  list(x = as.data.frame(lapply(mat$x, normalize)), y = normalize(mat$y))
})

# Function to train and evaluate models
train_and_evaluate <- function(x, y, hidden_layers, linear_output = FALSE) {
  # Create a formula based on the number of input variables
  formula <- as.formula(paste("y ~", paste(colnames(x), collapse = " + ")))

  # Train the model
  model <- neuralnet(formula, data = cbind(x, y = y), hidden = hidden_layers, linear.output = linear_output)

  # Make predictions
  predictions <- predict(model, x)

  # Calculate performance metrics
  rmse <- rmse(y, predictions)
  mae <- mae(y, predictions)
  mape <- mape(y, predictions)
  smape <- smape(y, predictions)

  return(list(model = model, metrics = c(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape)))
}

# Define different network structures
structures <- list(
  list(hidden = c(5), linear_output = FALSE),
  list(hidden = c(10), linear_output = FALSE),
  list(hidden = c(5, 5), linear_output = FALSE),
  list(hidden = c(10, 5), linear_output = FALSE)
)

# Train and evaluate models
results <- list()
for (i in seq_along(normalized_matrices)) {
  for (j in seq_along(structures)) {
    name <- paste0("Lag", lag_orders[i], "_", paste(structures[[j]]$hidden, collapse = "_"))
    results[[name]] <- train_and_evaluate(normalized_matrices[[i]]$x, normalized_matrices[[i]]$y,
                                          structures[[j]]$hidden, structures[[j]]$linear_output)
  }
}

# Create comparison table
comparison_table <- do.call(rbind, lapply(names(results), function(name) {
  c(name, unlist(results[[name]]$metrics),
    paste(lag_orders[as.numeric(substr(name, 4, 4))], "lags,",
          gsub("_", " hidden nodes, ", name)))
}))

colnames(comparison_table) <- c("Model", "RMSE", "MAE", "MAPE", "sMAPE", "Description")
print(comparison_table)

# Function to calculate number of weights
calculate_weights <- function(input_size, hidden_layers, output_size = 1) {
  weights <- input_size * hidden_layers[1] + hidden_layers[1]
  for (i in 2:length(hidden_layers)) {
    weights <- weights + hidden_layers[i-1] * hidden_layers[i] + hidden_layers[i]
  }
  weights <- weights + hidden_layers[length(hidden_layers)] * output_size + output_size
  return(weights)
}

# Find best one-hidden and two-hidden layer models
best_one_hidden <- names(results)[which.min(sapply(results[grep("_[0-9]+$", names(results))], function(x) x$metrics["RMSE"]))]
best_two_hidden <- names(results)[which.min(sapply(results[grep("_[0-9]+_[0-9]+$", names(results))], function(x) x$metrics["RMSE"]))]

cat("Best one-hidden layer model:", best_one_hidden, "\n")
cat("Number of weights:", calculate_weights(lag_orders[as.numeric(substr(best_one_hidden, 4, 4))], as.numeric(strsplit(best_one_hidden, "_")[[1]][-1])), "\n\n")

cat("Best two-hidden layer model:", best_two_hidden, "\n")
cat("Number of weights:", calculate_weights(lag_orders[as.numeric(substr(best_two_hidden, 4, 4))], as.numeric(strsplit(best_two_hidden, "_")[[1]][-1])), "\n")

# Evaluate best model on test set
best_model_name <- names(results)[which.min(sapply(results, function(x) x$metrics["RMSE"]))]
best_model <- results[[best_model_name]]

# Make predictions on the test set
# lag_order <- lag_orders[as.numeric(substr(best_model_name, 4, 4))]
lag_order <- 10
test_matrix <- create_io_matrix(test_data, lag_order)
test_x <- as.data.frame(lapply(test_matrix$x, normalize))
test_y <- normalize(test_matrix$y)

print("Debug: Final test_x column names")
print(colnames(test_x))

predictions <- predict(best_model$model, test_x)

print("Debug: Best model name")
print(best_model_name)

print("Debug: Test matrix column names")
print(colnames(test_x))

print("Debug: Model variables")
print(best_model$model$model.list$variables)


print("Debug: Final test_x column names")
print(colnames(test_x))

predictions <- predict(best_model$model, test_x)

# Denormalize function
denormalize <- function(x, orig_data) {
  return(x * (max(orig_data) - min(orig_data)) + min(orig_data))
}

actual <- denormalize(test_y, ts_data)
predicted <- denormalize(predictions, ts_data)

# Calculate performance metrics
test_metrics <- c(
  RMSE = rmse(actual, predicted),
  MAE = mae(actual, predicted),
  MAPE = mape(actual, predicted),
  sMAPE = smape(actual, predicted)
)

print(test_metrics)

# Plot the results
plot(actual, type = "l", col = "blue", xlab = "Time", ylab = "USD/EUR Exchange Rate",
     main = "Actual vs Predicted USD/EUR Exchange Rate")
lines(predicted, col = "red")
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1)

# Create a scatter plot
plot(actual, predicted, xlab = "Actual", ylab = "Predicted",
     main = "Scatter Plot: Actual vs Predicted USD/EUR Exchange Rate")
abline(a = 0, b = 1, col = "red", lty = 2)