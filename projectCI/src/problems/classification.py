import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class TitanicProblem:
    def __init__(self, data_path=None):
        self.data_path = r"D:\CI_Project\CI_Project\data\titanic.csv"
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None

    def load_and_preprocess(self):
        df = pd.read_csv(self.data_path) if self.data_path else self._get_sample_data()

        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

        for col in ['Sex', 'Embarked']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        X = df.drop('Survived', axis=1)
        y = df['Survived']

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42
        )

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def evaluate(self, y_true, y_pred, y_prob=None):
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        if y_prob is not None:
            metrics["AUC-ROC"] = roc_auc_score(y_true, y_prob)
        return metrics

    def _get_sample_data(self):
        data = {
            'Pclass': np.random.choice([1, 2, 3], 891),
            'Sex': np.random.choice(['male', 'female'], 891),
            'Age': np.random.randint(1, 80, 891),
            'SibSp': np.random.randint(0, 5, 891),
            'Parch': np.random.randint(0, 5, 891),
            'Fare': np.random.uniform(0, 500, 891),
            'Embarked': np.random.choice(['C', 'Q', 'S'], 891),
            'Survived': np.random.choice([0, 1], 891, p=[0.62, 0.38])
        }
        return pd.DataFrame(data)