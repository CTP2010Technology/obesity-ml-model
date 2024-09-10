import pandas as pd
import joblib

def main():
    # Load the trained model
    pipeline = joblib.load('obesity_model.pkl')

    # Load the example data
    data = pd.read_csv('examples/sample_data.csv')

    # Make predictions
    predictions = pipeline.predict(data)

    # Print predictions
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()
