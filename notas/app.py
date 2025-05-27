import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

colunas = ['Hours', 'Practice', 'TeamWork', 'MidTerm', 'FinalExam', 'Scores']
notas = pd.read_csv("notas.csv")

notas['Passou'] = notas['Grade'].apply(lambda x: 1 if x in ['A', 'B'] else 0)
feature_cols = ['Hours', 'Practice', 'TeamWork', 'MidTerm', 'FinalExam', 'Scores']
X = notas[feature_cols]
y = notas['Passou']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=16, stratify=y)

calculo = LogisticRegression(random_state=16)
calculo.fit(X_train, y_train)
y_pred = calculo.predict(X_test)

target_labels = [0, 1]
target_names = ['Reprovado', 'Aprovado']

cnf_matrix = metrics.confusion_matrix(y_test, y_pred, labels=target_labels)
#print("Matriz de Confusão:")
#print(cnf_matrix)

#print("\nRelatório de Classificação:")
#print(classification_report(y_test, y_pred, labels=target_labels, target_names=target_names))

print("Digite os dados do aluno")
hours = float(input("Horas estudadas: "))
practice = float(input("Horas prática: "))
teamwork= float(input("Horas trabalhadas em grupo: "))
midterm = float(input("Nota da primeira avaliação: "))
finalexam = float(input("Nota do exame final: "))
scores = float(input("Pontuação: "))

notas_novas = pd.DataFrame([{
    'Hours': hours,
    'Practice': practice,
    'TeamWork': teamwork, 
    'MidTerm': midterm,
    'FinalExam': finalexam,
    'Scores': scores,
}])

notas_novas_scaled = scaler.transform(notas_novas)

previsao = calculo.predict(notas_novas_scaled)
resultado = "Aprovado" if previsao[0] == 1 else "Reprovado"
print(f"resultado: {resultado}")




