import argparse
import pandas as pd
from sonar_model import load_sonar_raw, split_train_test, prepare_data_for_algorithm, get_model_registry, train_model_with_cv, create_cv_results_table, evaluate_binary_classifier
from sonar_model.config import DEFAULT_SEED, DEFAULT_CV_FOLDS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="../../data/raw/sonar.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--scoring", type=str, default="f2")
    p.add_argument("--fp-cost", type=float, default=1.0)
    p.add_argument("--fn-cost", type=float, default=5.0)
    args = p.parse_args()
    
    df = load_sonar_raw(path=args.data_path)
    X_train, X_test, y_train, y_test = split_train_test(df, test_size=args.test_size, seed=args.seed)
    prepared = prepare_data_for_algorithm(X_train, X_test, y_train, y_test)
    registry = get_model_registry(seed=args.seed)
    
    results = []
    for name, (estimator, param_grid) in registry.items():
        print(f"Training {name}...")
        try:
            res = train_model_with_cv(
                model=estimator, 
                param_grid=param_grid, 
                X_train=prepared.X_train, 
                y_train=prepared.y_train, 
                scoring=args.scoring, 
                cv_folds=args.cv_folds, 
                seed=args.seed,
                fp_cost=args.fp_cost,
                fn_cost=args.fn_cost
            )
            res.model_name = name
            results.append(res)
        except Exception as e:
            print(f"  FAILED: {str(e)[:80]}")
    
    leaderboard = create_cv_results_table(results, rank_by=f"cv_{args.scoring}_mean")
    
    print("\n" + "="*120)
    print("CV LEADERBOARD")
    print("="*120)
    
    print(f"{'Model':<20} {'F2':>8} {'±':>6} {'Recall':>8} {'±':>6} {'Prec':>8} {'±':>6} {'Acc':>8} {'Cost':>8}")
    print("-"*120)
    
    for _, row in leaderboard.iterrows():
        print(f"{row['model']:<20} "
              f"{row.get('cv_f2_mean', 0):.4f}  "
              f"{row.get('cv_f2_std', 0):.4f}  "
              f"{row.get('cv_recall_mean', 0):.4f}  "
              f"{row.get('cv_recall_std', 0):.4f}  "
              f"{row.get('cv_precision_mean', 0):.4f}  "
              f"{row.get('cv_precision_std', 0):.4f}  "
              f"{row.get('cv_accuracy_mean', 0):.4f}  "
              f"{row.get('cv_cost_mean', 0):.2f}")
    
    print("="*120)
    
    champ = None
    champ_name = str(leaderboard.iloc[0]["model"])
    for r in results:
        if r.model_name == champ_name:
            champ = r
            break
    
    if champ:
        print(f"\nCHAMPION: {champ.model_name}")
        print(f"Best Params: {champ.best_params}\n")
        
        test_metrics = evaluate_binary_classifier(
            champ.fitted_model, 
            prepared.X_test, 
            prepared.y_test,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost
        )
        
        print("Test Set Performance:")
        print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
        print(f"  Precision : {test_metrics['precision']:.4f}")
        print(f"  Recall    : {test_metrics['recall']:.4f}")
        print(f"  F1        : {test_metrics['f1']:.4f}")
        print(f"  F2        : {test_metrics['f2']:.4f}")
        print(f"  Cost      : {test_metrics['cost']:.2f}")
        if 'roc_auc' in test_metrics:
            print(f"  ROC AUC   : {test_metrics['roc_auc']:.4f}")
        if 'pr_auc' in test_metrics:
            print(f"  PR AUC    : {test_metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
