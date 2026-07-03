# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest classifier implemented with scikit-learn (RandomForestClassifier). It uses 100 estimators, a minimum of 5 samples per leaf, and a fixed random state of 42 for reproducibility. The model was developed as part of the Udacity Machine Learning DevOps Engineer Nanodegree to demonstrate a complete machine learning pipeline with continuous integration and cloud deployment. Categorical features are encoded with a one-hot encoder and the target label is binarised with a label binarizer, both fitted during training and saved alongside the model.

## Intended Use

The model predicts whether a person's annual income exceeds $50,000 based on demographic and employment attributes from census data. It is intended strictly for educational purposes, namely demonstrating model training, testing, API deployment, and CI/CD practices. It is not intended for use in any real-world decision-making context such as lending, hiring, or benefits eligibility, because it is trained on outdated census data and encodes historical societal biases.

## Training Data

The model was trained on the publicly available Census Income (Adult) dataset from the UCI Machine Learning Repository, containing 32,561 records with 14 features and a binary salary label (<=50K or >50K). The raw file contained extraneous spaces which were removed during cleaning, and the original hyphenated column names were preserved. Eighty percent of the data (26,048 records) was used for training, obtained through a stratified train-test split on the salary label with a fixed random state of 42. Eight categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country) were one-hot encoded, and the six continuous features were passed through unchanged.

## Evaluation Data

The remaining twenty percent of the data (6,513 records) was held out as a test set. The test data was processed with the encoder and label binarizer fitted on the training data, ensuring no information leakage from the evaluation set into training.

## Metrics

The model was evaluated using precision, recall, and F1 score (F-beta with beta equal to 1). On the held-out test set the model achieved a precision of 0.7847, a recall of 0.6135, and an F1 score of 0.6886. Performance was also computed on slices of the data for every unique value of each categorical feature, with the full results written to slice_output.txt. The slice analysis shows that performance varies across subgroups, for example across education levels and native countries, which is important context for assessing where the model is more or less reliable.

## Ethical Considerations

The dataset contains sensitive attributes including race, sex, and native country, and the model's predictions are correlated with these attributes. The slice metrics reveal unequal performance across demographic subgroups, meaning the model could systematically disadvantage certain groups if used for real decisions. The underlying data was collected in 1994 and reflects historical income distributions and societal biases of that period, so any patterns learned by the model should not be treated as representative of the present day.

## Caveats and Recommendations

The data is more than two decades old and should not be used to draw conclusions about current income distributions. The classes are imbalanced, with roughly 76 percent of records labelled <=50K, which contributes to the lower recall on the minority class. Several features contain missing values encoded as a question mark, which were retained as a distinct category. Future work could include hyperparameter tuning, class-imbalance mitigation, fairness-aware evaluation with a toolkit such as Aequitas, and periodic retraining on more recent data if a comparable modern dataset becomes available.
