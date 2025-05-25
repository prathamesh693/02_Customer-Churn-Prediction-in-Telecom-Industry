# Data Split Information

To train and evaluate the predictive models effectively, the dataset is split as follows:

- **Training set:** 80% of the data, used for model training and hyperparameter tuning.  
- **Test set:** 20% of the data, held out for final model evaluation to assess generalization performance.  
- **Stratified Split:** The split is stratified based on the 'Churn' target variable to maintain the same proportion of churned vs. non-churned customers in both training and test sets.  
- **Random state:** A fixed random seed (e.g., 42) is used to ensure reproducibility of the split.
