def registrar(nome, busca, X_te, y_te, duracao, n_candidatos):

    pipe = busca.best_estimator_
    teste = comum.avaliar_teste(pipe, X_te, y_te)
    res = busca.cv_results_
    i = busca.best_index_

    cv = {
        m: float(res[f"mean_test_{m}"][i])
        for m in comum.METRICAS
    }

    cv.update({
        f"{m}_std": float(res[f"std_test_{m}"][i])
        for m in comum.METRICAS
    })

    melhores = {
        k.replace("modelo__", ""): v
        for k, v in busca.best_params_.items()
    }

    comum.log(
        f"{nome}: "
        f"CV acc {cv['acuracia']*100:.2f} ± "
        f"{cv['acuracia_std']*100:.2f} | "
        f"Teste acc {teste['acuracia']*100:.2f} | "
        f"F1 {teste['f1']*100:.2f} | "
        f"AUC {teste['roc_auc']*100:.2f}"
    )

    return {
        "estrategia": nome,
        "melhores_parametros": melhores,
        "n_candidatos": int(n_candidatos),
        "tempo_s": float(duracao),
        "cv": cv,
        "teste": teste,
    }


## Função que gera a matriz de confusão##
def gerar_matriz_confusao(modelo, X_te, y_te):

    y_pred = modelo.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)

    return {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def gerar_roc(modelo, X_te, y_te):
    probs = modelo.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, probs)
    roc_auc = auc(fpr, tpr)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(roc_auc),
    }

def main():

    comum.log("Experimento Random Forest")

    # Ajusta o caminho dos dados no módulo comum
    comum.CAMINHO_DADOS = "adult.csv"
    df = comum.carregar_dados()
    X, y = comum.separar_xy(df)
    X_tr, X_te, y_tr, y_te = comum.dividir(X, y)
    cv = comum.validacao_cruzada()
    saida = {
        "experimento": "random_forest",
        "estrategias": []
    }


    # MODELO DE REFERÊNCIA
    PARAM_RF = {
        "n_estimators": 100,
        "max_depth": 7,
        "min_samples_leaf": 10,
        "min_samples_split": 20,
    }

    rf_inicial = comum.montar(
        RandomForestClassifier(
            **PARAM_RF,
            random_state=comum.SEED,
            n_jobs=N_JOBS
        ),
        X_tr
    )

    cv_ini = comum.avaliar_cv(
        rf_inicial,
        X_tr,
        y_tr,
        n_jobs=N_JOBS
    )

    rf_inicial.fit(X_tr, y_tr)

    saida["modelo_inicial"] = {
        "parametros": PARAM_RF,
        "cv": cv_ini,
        "teste": comum.avaliar_teste(
            rf_inicial,
            X_te,
            y_te
        )
    }


    ## CURVA DE VALIDAÇÃO
    n_estimators = [
        10,
        25,
        50,
        100,
        200
        #300,
        #500
    ]

    tr_scores, val_scores = validation_curve(
        rf_inicial,
        X_tr,
        y_tr,
        param_name="modelo__n_estimators",
        param_range=n_estimators,
        cv=cv,
        scoring="accuracy",
        n_jobs=N_JOBS
    )

    saida["curva_validacao"] = {
        "n_estimators": n_estimators,
        "train_accuracy":
            tr_scores.mean(axis=1).tolist(),
        "valid_accuracy":
            val_scores.mean(axis=1).tolist()
    }


    # GRID SEARCH

    grade = {

        "modelo__n_estimators":[100, 200],
        "modelo__max_depth":[5, 10, 20],
        "modelo__min_samples_leaf":[1, 2],
        "modelo__min_samples_split":[2, 5],
        "modelo__max_features":["sqrt", None],
        "modelo__class_weight":[None, "balanced"]
    }

    if RAPIDO:
        grade = {
            k: v[:2]
            for k, v in grade.items()
        }

    n_grade = int(
        np.prod([len(v) for v in grade.values()])
    )

    t0 = time.perf_counter()

    base = comum.montar(
        RandomForestClassifier(
            random_state=comum.SEED,
            n_jobs=N_JOBS
        ),
        X_tr
    )

    gs = GridSearchCV(
        base,
        grade,
        cv=cv,
        scoring=comum.METRICAS,
        refit="acuracia",
        n_jobs=N_JOBS
    )

    gs.fit(X_tr, y_tr)

    saida["estrategias"].append(
        registrar(
            "grid_search",
            gs,
            X_te,
            y_te,
            time.perf_counter() - t0,
            n_grade
        )
    )

    # RANDOM SEARCH

    distribuicoes = {
        "modelo__n_estimators": randint(50, 200),
        "modelo__max_depth": randint(3, 40),
        "modelo__min_samples_leaf": randint(1, 30),
        "modelo__min_samples_split": randint(2, 50),
        "modelo__max_features": ["sqrt", "log2", None],
        "modelo__class_weight": [None, "balanced"]
    }

    n_iter = 50 if not RAPIDO else 10

    t0 = time.perf_counter()

    rs = RandomizedSearchCV(
        base,
        distribuicoes,
        n_iter=n_iter,
        cv=cv,
        scoring=comum.METRICAS,
        refit="acuracia",
        n_jobs=N_JOBS,
        random_state=comum.SEED
    )

    rs.fit(X_tr, y_tr)

    saida["estrategias"].append(
        registrar(
            "random_search",
            rs,
            X_te,
            y_te,
            time.perf_counter() - t0,
            n_iter
        )
    )


    # MELHOR MODELO

    melhor = rs.best_estimator_
    melhor.fit(X_tr, y_tr)


    # MATRIZ DE CONFUSÃO

    saida["matriz_confusao"] = gerar_matriz_confusao(
        melhor,
        X_te,
        y_te
    )


    # ROC

    saida["roc_auc"] = gerar_roc(
        melhor,
        X_te,
        y_te
    )


    # FEATURE IMPORTANCE
    saida["importancias" ] = gerar_importancias(
        melhor
    )


    # SALVAR
    comum.salvar_resultado(
        "random_forest",
        saida
    )

    comum.log(
        "Concluído: resultados/random_forest.json"
    )


if __name__ == "__main__":
    main()