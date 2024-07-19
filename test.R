# Load necessary libraries
library(readxl)
library(neuralnet)
library(caret)
library(ggplot2)

# Load the dataset
data <- read_excel("C://Users//dlahi//OneDrive//Desktop//Assigments//ML ref def//ExchangeUSD.xlsx")
exchange_rates <- data$`USD/EUR`

# Split the data into training (first 400) and testing (last 100) sets
train_data <- exchange_rates[1:400]
test_data <- exchange_rates[401:500]

# Normalize the data
normalize <- function(x) { return((x - mean(x)) / sd(x)) }
train_data_norm <- normalize(train_data)
test_data_norm <- normalize(test_data)

# Function to create lagged data
create_lagged_data <- function(series, lag) {
  n <- length(series)
  X <- embed(series, lag + 1)
  X <- data.frame(X)
  colnames(X) <- c(paste0("Lag", lag:1), "Target")
  return(X)
}

# Create input/output matrices for different lags
lags <- 1:3
io_matrices <- lapply(lags, function(lag) { create_lagged_data(train_data_norm, lag) })

# Evaluate MLP models
evaluate_mlp <- function(io_matrix, hidden_layers) {
  set.seed(123)
  train_io <- io_matrix[1:(nrow(io_matrix) - 100), ]
  test_io <- io_matrix[(nrow(io_matrix) - 99):nrow(io_matrix), ]
  formula <- as.formula(paste("Target ~", paste(colnames(io_matrix)[1:(ncol(io_matrix)-1)], collapse = " + ")))
  nn <- neuralnet(formula, data = train_io, hidden = hidden_layers, linear.output = TRUE)
  test_pred <- compute(nn, test_io[, colnames(train_io)[-ncol(train_io)]])$net.result
  test_pred <- test_pred * sd(train_data) + mean(train_data)
  actual <- test_io$Target * sd(train_data) + mean(train_data)
  rmse <- sqrt(mean((test_pred - actual)^2))
  mae <- mean(abs(test_pred - actual))
  mape <- mean(abs((test_pred - actual) / actual)) * 100
  smape <- mean(2 * abs(test_pred - actual) / (abs(test_pred) + abs(actual))) * 100
  return(c(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}

# Evaluate different models
results <- data.frame()
for (lag in lags) {
  io_matrix <- io_matrices[[lag]]
  for (hidden in list(c(5), c(10), c(5, 5), c(10, 10))) {
    performance <- evaluate_mlp(io_matrix, hidden)
    results <- rbind(results, c(Lag = lag, Hidden = paste(hidden, collapse = "-"), performance))
  }
}

# Add column names
colnames(results) <- c("Lag", "Hidden Layers", "RMSE", "MAE", "MAPE", "sMAPE")
print(results)

# Find the best model
best_model <- results[which.min(results$RMSE), ]
print(best_model)

# Graphical representation of best model's results
best_lag <- best_model$Lag
best_hidden <- as.numeric(strsplit(as.character(best_model$`Hidden Layers`), "-")[[1]])
best_io_matrix <- io_matrices[[best_lag]]
train_io <- best_io_matrix[1:(nrow(best_io_matrix) - 100), ]
test_io <- best_io_matrix[(nrow(best_io_matrix) - 99):nrow(best_io_matrix), ]
best_nn <- neuralnet(formula, data = train_io, hidden = best_hidden, linear.output = TRUE)
test_pred <- compute(best_nn, test_io[, colnames(train_io)[-ncol(train_io)]])$net.result
test_pred <- test_pred * sd(train_data) + mean(train_data)
actual <- test_io$Target * sd(train_data) + mean(train_data)
plot_df <- data.frame(Date = as.Date(data$`YYYY/MM/DD`[401:500]), Actual = actual, Predicted = test_pred)
ggplot(plot_df, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "Exchange Rate Forecasting: Actual vs Predicted", y = "Exchange Rate (USD/EUR)", color = "Legend") +
  theme_minimal()
