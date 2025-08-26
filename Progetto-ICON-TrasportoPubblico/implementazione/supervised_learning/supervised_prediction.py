"""
Modulo per l'apprendimento supervisionato sul dataset dei treni.

Include:
- suddivisione train/test
- validazione incrociata KFold solo sul training
- calcolo di R^2 con deviazione standard
- learning curve
- esempio di predizione con il modello addestrato

Autore: Francesco Moreo
Data: 26/08/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

def run_supervised_learning(X, y, test_size=0.2, random_state=42, **model_kwargs):
    """
    Esegue apprendimento supervisionato con Random Forest:
    - train/test split
    - KFold CV sul training
    - learning curve
    - valutazione su test
    - esempio di predizione

    Args:
        X (pd.DataFrame or np.ndarray): features
        y (pd.Series or np.ndarray): target
        test_size (float): proporzione test set
        random_state (int): seme per la riproducibilità
        **model_kwargs: parametri aggiuntivi per RandomForestRegressor

    Returns:
        None
    """

    # Suddivisione train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Modello
    model = RandomForestRegressor(random_state=random_state, **model_kwargs)

    # KFold CV solo sul training
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

    print("=== Validazione incrociata (solo training) ===")
    print(f"R^2 medio (CV): {cv_scores.mean():.4f}")
    print(f"Deviazione standard (CV): {cv_scores.std():.4f}\n")

    # Addestramento finale sul training
    model.fit(X_train, y_train)

    # Valutazione su test
    y_pred_test = model.predict(X_test)
    test_score = r2_score(y_test, y_pred_test)
    print("=== Valutazione su test set ===")
    print(f"R^2 (test): {test_score:.4f}\n")

    # Learning curve
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, marker='o', label="Training score")
    plt.plot(train_sizes, valid_mean, marker='s', label="Validation score")
    plt.xlabel("Numero di campioni di training")
    plt.ylabel("R^2")
    plt.title("Learning Curve (Random Forest)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Esempio di predizione con il modello finale
    if hasattr(X_test, "iloc"):
        example_input = X_test.iloc[0].values[np.newaxis, :]
        real_value = y_test.iloc[0]
    else:
        example_input = np.array(X_test[0])[np.newaxis, :]
        real_value = y_test[0]

    prediction = model.predict(example_input)

    print("=== Esempio di predizione ===")
    print(f"Input: {example_input}")
    print(f"Predizione: {prediction[0]:.4f}")
    print(f"Valore reale: {real_value}")
