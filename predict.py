import argparse

import joblib
from sklearn.datasets import load_iris

from train import MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Predict an iris species from four measurments in cm"
            )

    parser.add_argument("sepal_length", type=float)
    parser.add_argument("sepal_width", type=float)
    parser.add_argument("petal_length", type=float)
    parser.add_argument("petal_width", type=float)
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        parser.error("run python train.py first")


    measurments = [
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width,

        ]

    model = joblib.load(MODEL_PATH)
    class_id = model.predict([measurments])[0]
    species = load_iris().target_names[class_id]
    print(f"Predicted species: {species}")


if __name__ == "__main__":
    main()



