import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


# loading data from diabetes.csv
colunas = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi','pedigree','age','label']
pima = pd.read_csv("diabetes.csv")
pima.head()

#split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
X = pima[feature_cols] 
y = pima['Outcome'] 

# divide the dataset into training set and test set
# parameters: features, target, test_set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=16)

# create a logic regression classifier object
# fit() fit your model on the train set
# predict() perform prediction on the test set

# instantiate the model
logreg = LogisticRegression(random_state=16)

# fit the model
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# model evaluation
# evaluate the perfomance of a classification model

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# classification report for accuracy, precision and recall
target_names = ['sem diabetes', 'com diabetes']
# print(classification_report(y_test, y_pred, target_names=target_names))

"""
INPUT => DADOS NOVOS
"""

print("Digite os dados do paciente para diagnóstico")

pregnancy = float(input("Número de gestações: "))
insulin = float(input("Nível de insulina: "))
glucose = float(input("Glicose: "))
bp = float(input("Pressão sanguínea: "))
bmi = float(input("Índice de massa corporal (BMI): "))
pedigree = float(input("Histórico genético (pedigree): "))
age = float(input("Idade: "))

dados_novos = pd.DataFrame([{
    'Pregnancies': pregnancy,
    'Insulin': insulin,
    'BMI': bmi,
    'Age': age,
    'Glucose': glucose,
    'BloodPressure': bp,
    'DiabetesPedigreeFunction': pedigree,
}])

# comparar com novos dados
dados_novos_scaled = scaler.transform(dados_novos)

previsao = logreg.predict(dados_novos_scaled)
resultado = "Com diabetes" if previsao[0] == 1 else "Sem diabetes"
print(f"\nDiagnóstico: {resultado}")