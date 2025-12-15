import joblib
from pathlib import Path
from sonar_model import load_sonar_raw, split_train_test, prepare_data_for_algorithm, get_model_registry, train_model_with_cv
from sonar_model.config import DEFAULT_SEED

def save_champion_model():
    df = load_sonar_raw("../../data/raw/sonar.csv")
    X_train, X_test, y_train, y_test = split_train_test(df, seed=DEFAULT_SEED)
    prepared = prepare_data_for_algorithm(X_train, X_test, y_train, y_test)
    
    registry = get_model_registry(seed=DEFAULT_SEED)
    svm_model, svm_params = registry["svm"]
    
    result = train_model_with_cv(
        model=svm_model,
        param_grid=svm_params,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        scoring="f2",
        seed=DEFAULT_SEED
    )
    
    model_path = Path("sonar_model/trained_models/champion_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(result.fitted_model, model_path)
    
    api_model_path = Path("../sonar_api/sonar_api/trained_model.pkl")
    joblib.dump(result.fitted_model, api_model_path)
    
    print(f"Model saved to {model_path}")
    print(f"Model copied to {api_model_path}")
    print(f"Model type: {type(result.fitted_model).__name__}")

if __name__ == "__main__":
    save_champion_model()
