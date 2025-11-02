CMSE492 Project

Title:
Predicting NCAA Tournament Success from Team Efficiency Metrics

Description:
College basketball statistics have become increasingly advanced, with metrics like adjusted offensive and defensive efficiency (ADJOE, ADJDE), effective field goal percentage (EFG), turnover rate (TOR), and tempo (ADJ_T) offering a deeper look into how well a team performs. While these numbers help describe team strength, it is still difficult to accurately predict how a team will perform in the NCAA Tournament.

This project will use machine learning to predict NCAA Tournament outcomes — such as whether a team makes the tournament, what seed it receives, or how far it advances — based on regular-season efficiency data. The project will include data exploration, feature engineering (like calculating efficiency margins or tempo-adjusted stats), and model development using methods such as logistic regression, random forests, and XGBoost. These models will be compared to see which performs best and which features are the most important for predicting tournament success.

The project directory will include a README.md, .gitignore, a data folder (with raw and cleaned datasets), a notebooks folder for exploratory Jupyter notebooks, and an src folder for code that handles data processing, model training, and evaluation. There will also be folders for figures (visuals and plots), docs (reports and notes), and a requirements.txt file listing dependencies.

The project will be built using Python and Jupyter Notebook, with libraries such as pandas, NumPy, scikit-learn, XGBoost, and Matplotlib. Future work could include using neural networks or running simulated tournament brackets based on model predictions.