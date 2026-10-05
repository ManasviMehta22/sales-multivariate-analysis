Multivariate Analysis: 
LDA/QDA & Factor Analysis on Sales Data

Project Overview
This project applies multivariate statistical techniques (Factor Analysis, Linear and Quadratic Discriminant Analysis) in R to analyze a retail sales dataset. The objective is to uncover latent data structures and classify product categories (Menswear vs. Womenswear) based on geographic and transactional attributes.   

Dataset
File: 
Sales_Product_Details.csv   
Numeric Features: 
Quantity, Unit Price, Sales Revenue, Latitude, and Longitude.   
Preprocessing: 
Log transformation was applied to Unit Price to handle skewness, followed by Z-score standardization across all numeric variables to maximize classification accuracy.   

Methodology & Key Findings
1. Factor Analysis (Dimensionality Reduction)
Parallel analysis and a scree plot were utilized to determine the optimal number of factors. A two-factor varimax rotation solution effectively explained the dataset's structure:
  Factor 1 (Sales Volume): Heavily dominated by Sales Revenue and Quantity, accounting for 53% of the explained variance.
  Factor 2 (Geography): Defined by Latitude and Longitude, accounting for 47% of the explained variance.
2. Classification (LDA & QDA)
Linear and Quadratic Discriminant Analysis models were trained to predict the product category.
Model Performance:
The LDA model achieved 69.23% accuracy, while the QDA model achieved 73.08% accuracy. Both models yielded an AUC of 0.73.
Algorithmic Comparison:
QDA outperformed LDA, demonstrating that relaxing the assumption of equal covariance matrices allowed for a more flexible and accurate curved decision boundary.
Feature Importance:
Unit Price and Latitude were the strongest discriminating variables. Menswear transactions generally associated with lower unit prices and higher latitudes, whereas Womenswear correlated with higher unit prices and lower latitudes.

Technologies & Libraries
Language: R
Libraries: psych, corrplot, ggplot2, reshape2, MASS, tidyverse, caret, pROC, pacman. 
