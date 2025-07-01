import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Carregar os dados
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")
print("Primeiras linhas do treino:")
print(df_train.head(5))

# 2. Remover colunas desnecessárias
drop_cols = ['Name', 'Ticket', 'Cabin']
df_train.drop(columns=drop_cols, inplace=True)
df_test.drop(columns=drop_cols, inplace=True)

# 3. Verificar valores nulos
print("\nValores nulos antes do dropna:")
print(df_train.isnull().sum())

# 4. Remover valores nulos do treino
df_train.dropna(inplace=True)

# 5. Codificar colunas categóricas
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df_train['Sex'] = le_sex.fit_transform(df_train['Sex'])
df_train['Embarked'] = le_embarked.fit_transform(df_train['Embarked'])

# Preencher nulos do teste (sem perder dados)
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].mode()[0])

# Codificar no teste com o mesmo LabelEncoder do treino
df_test['Sex'] = le_sex.transform(df_test['Sex'])
df_test['Embarked'] = le_embarked.transform(df_test['Embarked'])

# 6. Análise visual
plt.figure(figsize=(10, 6))
sns.heatmap(df_train.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Map")
plt.show()

sns.countplot(x='Survived', hue='Sex', data=df_train)
plt.title("Survival by Sex")
plt.legend(["Female", "Male"])
plt.show()

sns.histplot(data=df_train, x='Fare', hue='Survived', bins=30, kde=True, element='step')
plt.title("Fare distribution by Survival")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df_train)
plt.title("Survival by Class")
plt.xlabel("Classe")
plt.ylabel("Número de Passageiros")
plt.legend(title="Sobreviveu", labels=["Não", "Sim"])
plt.show()

# 7. Treinamento do modelo
X = df_train.drop(["Survived", "PassengerId"], axis=1)
y = df_train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("\nAcurácia no conjunto de validação:", accuracy_score(y_val, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_val, y_pred))

# 8. Geração da submissão
X_test = df_test.drop("PassengerId", axis=1)
y_test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": y_test_pred
})

submission.to_csv("submission.csv", index=False)
print("\nArquivo 'submission.csv' gerado com sucesso!")
