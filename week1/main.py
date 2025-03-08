import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = "AI_class/iris.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 데이터 확인
print(df.head())  # 처음 5개 행 출력
print(df.columns)  # 컬럼명 출력

# 특성과 타겟(label) 분리
X = df.iloc[:, :-1]  # 마지막 열을 제외한 모든 열 (특성)
y = df.iloc[:, -1]   # 마지막 열 (타겟 값)

# 타겟(Label) 인코딩 (문자 → 숫자 변환)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터셋 분리 (80% 훈련, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 각 모델 초기화
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

# 모델 학습 및 평가
for name, model in models.items():
    model.fit(X_train, y_train)  # 학습
    y_pred = model.predict(X_test)  # 예측
    accuracy = accuracy_score(y_test, y_pred)  # 정확도 계산
    print(f"{name} Accuracy: {accuracy:.4f}")  # 정확도 출력