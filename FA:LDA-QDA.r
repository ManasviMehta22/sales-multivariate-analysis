# FACTOR ANALYSIS

# Load necessary libraries
# install.packages(c("psych", "corrplot", "ggplot2", "reshape2"))
library(psych) 
library(corrplot)
library(ggplot2)
library(reshape2)

# Load the dataset
data <- read.csv("~/Desktop/MSc DS & AI/Manasvi/Sales_Product_Details.csv")

# Preprocessing
# Filter for numeric columns only 
numeric_cols <- c("Quantity", "Unit_Price", "Sales_Revenue", "Latitude", "Longitude")
data_numeric <- data[, numeric_cols]

data_clean <- na.omit(data_numeric)

data_scaled <- scale(data_clean)

# Correlation Matrix
cor_matrix <- cor(data_scaled)
print("Correlation Matrix:")
print(cor_matrix)

# Determine Number of Factors
# Scree Plot
fa.parallel(data_scaled, n.obs = nrow(data_scaled), fa = "fa", main = "Scree Plot")

# Factor Analysis
# Rotation 'varimax' used to make loadings more interpretable.
fa_result <- fa(data_scaled, nfactors = 2, rotate = "varimax")
print(fa_result)

# Visualization of Results

# Factor Loadings Heatmap
loadings <- as.data.frame(unclass(fa_result$loadings))
loadings$Variable <- rownames(loadings)
loadings_m <- melt(loadings, id.vars = "Variable")

ggplot(loadings_m, aes(x = variable, y = Variable, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "lavender", high = "lightblue", mid = "#FFCCBC", midpoint = 0) +
  theme_minimal() +
  labs(title = "Factor Loadings Heatmap", x = "Factors", y = "Variables", fill = "Loading")

# LDA QDA

# 1. CLEANING, FILTERING, SCALING, AND STANDARDIZING
if (!require("pacman")) install.packages("pacman")
pacman::p_load(MASS, tidyverse, caret, pROC, reshape2, ggplot2)

data <- read.csv("~/Desktop/MSc DS & AI/Manasvi/Sales_Product_Details.csv")

# Filter for the two target categories only to maximize classification power
target_classes <- c("Menswear", "Womenswear")

final_data <- data %>%
  filter(Product_Category %in% target_classes) %>%
  mutate(Outcome = factor(Product_Category)) %>%
  # Feature Engineering: Log transform price to handle skewness
  mutate(Unit_Price = log1p(as.numeric(Unit_Price)),
         Quantity = as.numeric(Quantity),
         Latitude = as.numeric(Latitude),
         Longitude = as.numeric(Longitude)) %>%
  dplyr::select(Outcome, Quantity, Unit_Price, Latitude, Longitude) %>%
  na.omit()

# Standardization (Z-score scaling) - Mandatory for Kappa/Accuracy boost
num_cols <- c("Quantity", "Unit_Price", "Latitude", "Longitude")
final_data[num_cols] <- scale(final_data[num_cols])

# 2. PRINTING FINAL CLEANED DATA
cat("Final Cleaned and Scaled Data Summary\n")
print(head(final_data))
print(summary(final_data))

# 3. BOXPLOT OF VARIABLES
# Melting data for a cleaner ggplot boxplot
melted_data <- melt(final_data, id.vars = "Outcome")
ggplot(melted_data, aes(x = variable, y = value, fill = Outcome)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Boxplot of Standardized Variables", y = "Z-Score")

# 4. MODEL OF LDA AND QDA
model_lda <- lda(Outcome ~ ., data = final_data)
model_qda <- qda(Outcome ~ ., data = final_data)

cat("\n LDA Model Output \n")
print(model_lda)

cat("\n QDA Model Output \n")
print(model_qda)

# 5. PREDICTIONS
lda_pred <- predict(model_lda)
qda_pred <- predict(model_qda)

# 6. STANDARD LDA PLOT
plot(model_lda, col = as.numeric(final_data$Outcome), main = "Standard LDA Projection")

# 7. LDA SCATTER PLOT
lda_df <- data.frame(LD1 = lda_pred$x[,1], Outcome = final_data$Outcome)
ggplot(lda_df, aes(x = LD1, y = Outcome, color = Outcome)) +
  geom_jitter(width = 0, height = 0.2, alpha = 0.6) +
  theme_minimal() +
  labs(title = "LDA Scatter Plot: Class Separation")

# 8. GGPLOT DENSITY PLOT 
ggplot(lda_df, aes(x = LD1, fill = Outcome)) +
  geom_density(alpha = 0.6) +
  theme_minimal() +
  labs(title = "LDA Density Plot", x = "Discriminant Score (LD1)")

# 9. ROC COMPARISON (Both curves, TPR vs FPR)
roc_menswear <- roc(ifelse(final_data$Outcome == "Menswear", 1, 0), lda_pred$posterior[, "Menswear"])
roc_womenswear <- roc(ifelse(final_data$Outcome == "Womenswear", 1, 0), lda_pred$posterior[, "Womenswear"])

plot(roc_menswear, col = "steelblue", lwd = 3, legacy.axes = TRUE, 
     xlab = "False Positive Rate (FPR)", ylab = "True Positive Rate (TPR)",
     main = "ROC Comparison: Menswear vs Womenswear")
plot(roc_womenswear, col = "#FFB3BA", lwd = 3, add = TRUE)
legend("bottomright", legend = c(paste("Menswear AUC:", round(auc(roc_menswear), 2)),
                                 paste("Womenswear AUC:", round(auc(roc_womenswear), 2))),
       col = c("steelblue", "#FFB3BA"), lwd = 3)

# 10. CONFUSION MATRIX HEATMAP
cm_data <- as.data.frame(table(Predicted = lda_pred$class, Actual = final_data$Outcome))
ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradientn(colors = c("#E0F2F1", "#B2DFDB", "#FFCCBC", "#FFAB91")) +
  theme_minimal() + labs(title = "LDA Confusion Matrix Heatmap")

# 11. FINAL PERFORMANCE PRINTING
cat("\n LDA Performance \n")
print(cm)

cat("\n QDA Performance \n")
print(confusionMatrix(qda_pred$class, final_data$Outcome))