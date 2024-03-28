# Inventory Risk Prediction Model for Paccar Parts

## Description:
This project centers on the development of a predictive model aimed at optimizing order fulfillment processes by proactively identifying inventory items at risk of depletion. This model employs advanced machine learning techniques to prioritize order suggestions by classifying the inventory items based on their risk level, ultimately minimizing the likelihood of stockouts and enhancing operational efficiency.

## Key Features:
1. Proactive Risk Identification: Utilizes historical inventory data to identify items prone to depletion before stockouts occur.
2. Data Preprocessing and Feature Engineering: Implements robust data preprocessing techniques to clean and prepare the dataset for analysis. This includes one-hot encoding, normalization, and Principal Component Analysis (PCA) to handle categorical variables, standardize numerical features, and address high dimensionality, respectively.
3. Advanced Machine Learning Techniques: Leverages sophisticated algorithms, including Random Forest, to analyze and classify inventory items based on their depletion risk.
4. Data-Driven Decision Making: Considers various factors such as product demand, lead time, safety stock, on-hand quantity, and historical sales data to generate accurate predictions.
5. Optimized Order Prioritization: Prioritizes high-risk order suggestions, ensuring timely replenishment and reducing the risk of stockouts.
6. Operational Efficiency: Automates order prioritization to streamline fulfillment processes and mitigate the impact of regional disruptions.

## Result:
| Metric    | Training Set | Holdout Set |
|-----------|--------------|-------------|
| Accuracy  | 81%          | 79%         |
| Precision | 80%          | 74%         |

Our model achieved an overall accuracy of 79% in predicting high-risk inventory items., closely matching PACCAR's current in-production model at 80% accuracy. In conclusion, our analysis provide actionable insights to streamline order fulfillment processes and improve overall operational efficiency.

[PACCAR Parts Presentation.pdf](https://github.com/srimallipudi/Inventory-Risk-Prediction-Model-for-PACCAR-Parts/files/14794244/PACCAR.Parts.Presentation.pdf)
