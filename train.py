from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

MODEL_PATH= Path(__file__).resolve().parent / "artifacts" / "model.joblib"


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
            iris.data,
            iris.target,
            test_size=0.25,
            random_state=42,
            stratify=iris.target,
            )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    print(f"Test accuracy: {model.score(X_test, y_test):.1%}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()


