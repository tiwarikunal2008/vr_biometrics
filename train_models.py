import json, os
import numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

FEATURES = ["typing_speed_wpm", "dwell_ms", "flight_ms", "grip_pressure", "trigger_force", "hand_velocity", "orientation_deg"]

GENUINE  = np.array([(48, 5), (95, 10), (140, 18), (0.62, 0.05), (0.55, 0.06), (1.10, 0.12), (32, 4)])
IMPOSTOR = np.array([(40, 14), (115, 28), (170, 45), (0.50, 0.15), (0.48, 0.16), (1.00, 0.35), (38, 14)])

T = 10 
rng = np.random.default_rng(42)

def make(n, params):
    base = rng.normal(params[:, 0], params[:, 1], size=(n, len(FEATURES)))
    seq = base[:, None, :] + rng.normal(0, params[:, 1] * 0.25, size=(n, T, len(FEATURES)))
    return seq

N = 3000
seq = np.concatenate([make(N, GENUINE), make(N, IMPOSTOR)])
y = np.array([1] * N + [0] * N)
X = seq.mean(axis=1) 

Xtr, Xte, Str, Ste, ytr, yte = train_test_split(X, seq, y, test_size=0.25, stratify=y, random_state=1)
scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

results = {}
def record(name, pred):
    results[name] = {
        "accuracy": round(accuracy_score(yte, pred), 4),
        "f1": round(f1_score(yte, pred), 4)
    }
    print(name, results[name])

svm = SVC(kernel="rbf", probability=True, random_state=1).fit(Xtr_s, ytr)
joblib.dump(svm, "models/svm.pkl")
record("SVM", svm.predict(Xte_s))

rf = RandomForestClassifier(n_estimators=200, random_state=1).fit(Xtr_s, ytr)
joblib.dump(rf, "models/rf.pkl")
record("Random Forest", rf.predict(Xte_s))

try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss").fit(Xtr_s, ytr)
    joblib.dump(xgb, "models/xgb.pkl")
    record("XGBoost", xgb.predict(Xte_s))
except ImportError:
    print("xgboost not installed - skipped")


try:
    from tensorflow import keras 
    flat = lambda s: scaler.transform(s.reshape(-1, len(FEATURES))).reshape(s.shape)
    lstm = keras.Sequential([
        keras.layers.Input((T, len(FEATURES))),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    lstm.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    lstm.fit(flat(Str), ytr, epochs=8, batch_size=64, validation_split=0.1, verbose=0)
    lstm.save("models/lstm.keras")
    record("LSTM", (lstm.predict(flat(Ste), verbose=0).ravel() > 0.5).astype(int))
except ImportError:
    print("tensorflow not installed - LSTM skipped")

json.dump(results, open("models/metrics.json", "w"), indent=2)
print("Saved models/ and metrics.json")