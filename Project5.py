import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Зчитуємо дані з файлу train
with open('train_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()

# Розділяємо дані на текст відгуку і мітку настрою
X = [line.split(',', 1)[1].strip() for line in data]
y = [line.split(',')[0].strip() for line in data]

# Векторизуємо текст за допомогою TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Зчитуємо дані з файлу test
with open('test_data.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()

# Розділяємо дані на текст відгуку і мітку настрою
X_test = [line.split(',', 1)[1].strip() for line in data]
y_test = [line.split(',')[0].strip() for line in data]

# Векторизуємо тестові дані
X_tfidf_test = vectorizer.transform(X_test)


@st.cache_resource
def train_GradientBoosting():
    # Ініціалізуємо та навчаємо модель градієнтного бустингу
    gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=2, random_state=40)
    gb_classifier.fit(X_tfidf, y)

    # Прогнозуємо на тестових даних
    y_pred = gb_classifier.predict(X_tfidf_test)

    # Оцінюємо точність моделі
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy GradientBoosting:", accuracy)

    return gb_classifier


@st.cache_resource
def train_LightGBM():
    # Ініціалізуємо та навчаємо модель LightGBM
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=2, min_child_samples=1, min_data_in_bin=1, random_state=40)
    gbm.fit(X_tfidf, y)

    # Прогнозуємо на тестових даних
    y_pred = gbm.predict(X_tfidf_test)

    # Оцінюємо точність моделі
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy LightGBM:", accuracy)

    return gbm


# Перекодуємо мітки настрою в числові значення
label_encoder = LabelEncoder()
y_encode = label_encoder.fit_transform(y)

@st.cache_resource
def train_XGBoost():
    dtrain = xgb.DMatrix(X_tfidf, label=y_encode)
    # Ініціалізуємо та навчаємо модель XGBoost
    params = {
        'objective': 'binary:logistic',  # Використовується для задач бінарної класифікації
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 10
    }
    bst = xgb.train(params, dtrain, num_boost_round=500)

    # Перекодуємо мітки настрою в числові значення
    y_test_encoded = label_encoder.transform(y_test)
    # Створимо необхідний тип DMatrix для XGBoost
    dtest = xgb.DMatrix(X_tfidf_test, label=y_test_encoded)
    # Прогнозуємо на тестових даних
    y_pred_prob = bst.predict(dtest)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

    # Оцінюємо точність моделі
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print("Accuracy XGBoost:", accuracy)

    return bst


@st.cache_resource
def train_RandomForest():
    # Ініціалізуємо та навчаємо модель Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=45)
    rf_classifier.fit(X_tfidf, y)

    # Прогнозуємо на тестових даних
    y_pred = rf_classifier.predict(X_tfidf_test)

    # Оцінюємо точність моделі
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Random Forest:", accuracy)

    return rf_classifier


@st.cache_resource
def train_RandomForestRegressor():
    # Ініціалізуємо та навчаємо модель Random Forest
    rfr_classifier = RandomForestRegressor(n_estimators=300, max_depth=3,  random_state=45)
    rfr_classifier.fit(X_tfidf, y_encode)

    # Прогнозуємо на тестових даних
    y_pred_prob = rfr_classifier.predict(X_tfidf_test)
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

    # Перекодуємо мітки настрою в числові значення
    y_test_encoded = label_encoder.transform(y_test)

    # Оцінюємо точність моделі
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print("Accuracy Random Forest Regressor:", accuracy)

    return rfr_classifier


# Навчаємо моделі
gb_classifier = train_GradientBoosting()
gbm = train_LightGBM()
bst = train_XGBoost()
rf_classifier = train_RandomForest()
rfr_classifier = train_RandomForestRegressor()


# Функція визначення настрою окремою моделлю та загалом
def predict_sentiment(selected_model, user_input):
    # Векторизуємо користувацький ввід
    user_input_tfidf = vectorizer.transform([user_input])

    # Виконуємо визначення
    rf_pred = rf_classifier.predict(user_input_tfidf)[0]
    gb_pred = gb_classifier.predict(user_input_tfidf)[0]
    gbm_pred = gbm.predict(user_input_tfidf)[0]

    # Для Random Forest Regressor та XGBoost здійснюємо необхідну обробку вхідних та вихідних даних
    dtest_user = xgb.DMatrix(user_input_tfidf)
    bst_pred_prob = bst.predict(dtest_user)
    bst_pred = label_encoder.inverse_transform([1 if prob > 0.5 else 0 for prob in bst_pred_prob])[0]
    rfr_pred_prob = rfr_classifier.predict(user_input_tfidf)
    rfr_pred = label_encoder.inverse_transform([1 if prob > 0.5 else 0 for prob in rfr_pred_prob])[0]

    predictions = {
        "Random Forest": rf_pred,
        "Random Forest Regressor": rfr_pred,
        "GradientBoosting": gb_pred,
        "LightGBM": gbm_pred,
        "XGBoost": bst_pred
    }

    # Обчислюємо кількість Позитивних та Негативних значень
    counts = {"Позитивний": 0, "Негативний": 0}
    for pred in predictions.values():
        counts[pred] += 1
    # Визначаємо загальний настрій відгуку (за всіма моделями)
    total_prediction = "Позитивний" if counts["Позитивний"] > counts["Негативний"] else "Негативний"

    # Визначаємо вибрану модель
    if selected_model == "Random Forest":
        prediction = rf_pred
    elif selected_model == "Random Forest Regressor":
        prediction = rfr_pred
    elif selected_model == "GradientBoosting":
        prediction = gb_pred
    elif selected_model == "LightGBM":
        prediction = gbm_pred
    else:
        prediction = bst_pred
    return prediction, total_prediction


# Створюємо Streamlit додаток
st.title("Моделі аналізу настрою відгуків")

# Розділяємо інтерфейс на 2 колонки
col1, col2 = st.columns(2)

# Заповнюємо кожну з колонок: 2 відповідає за обрання моделі, 1 - за введення відгуку та виведення його настрою
with col2:
    select_model = st.radio("Виберіть модель: 👇", ["Random Forest", "GradientBoosting", "LightGBM", "XGBoost", "Random Forest Regressor"])

with col1:
    user_input = st.text_input("Введіть відгук: ")
    if user_input:
        sentiment, total_sentiment = predict_sentiment(select_model, user_input)
        st.write(f"Моделлю {select_model} відгук визначений як: {sentiment}")
        st.write(f"Загально відгук визначений як: {total_sentiment}")


