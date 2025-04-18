from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def tfidf_tuner(X, y, model_name='svc'):
    # Dictionary of models
    models = {
        'svc': SVC(),
        'logreg': LogisticRegression()
    }

    # Check if model name is valid
    if model_name not in models:
        raise ValueError(f"Model {model_name} is not supported. Choose from 'svc', 'logreg'.")

    # Select the model
    model = models[model_name]

    # Create a pipeline with TF-IDF and the selected model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        (model_name, model)
    ])

    # Define parameter grid for each model
    param_grid = {
        'tfidf__max_features': [200, 210, 230, 270, 330, 410, 510],
        'tfidf__ngram_range': [(1, 3), (1, 4)],
    }

    if model_name == 'svc':
        param_grid['svc__kernel'] = ['linear', 'poly', 'rbf']
    elif model_name == 'logreg':
        param_grid['logreg__C'] = [0.01, 0.1, 1, 10]
        param_grid['logreg__solver'] = ['saga', 'liblinear', 'lbfgs']

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=48)

    # Grid search
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X, y)

    return grid.best_params_



def genetic_algorithm_xgb_with_tfidf(text_data, y, params_bound=None, number_of_generations=10, number_of_population=50):
    if params_bound:
        PARAM_BOUNDS = params_bound
    else:
        PARAM_BOUNDS = {
            'n_estimators': (50, 500),
            'learning_rate': (0.001, 0.3),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0, 10),
            'max_features': (100, 10000),       # For TF-IDF
            'ngram_range': (1, 5)               # Will be used as (1, n)
        }

    # Create individual
    def create_individual():
        return [
            random.randint(*PARAM_BOUNDS['n_estimators']),
            random.uniform(*PARAM_BOUNDS['learning_rate']),
            random.uniform(*PARAM_BOUNDS['subsample']),
            random.uniform(*PARAM_BOUNDS['colsample_bytree']),
            random.uniform(*PARAM_BOUNDS['reg_alpha']),
            random.randint(*PARAM_BOUNDS['max_features']),
            random.randint(*PARAM_BOUNDS['ngram_range']),
        ]

    # Evaluation with cross-validation
    def evaluate_individual(individual, text_data, y):
        xgb_params = {
            'n_estimators': int(individual[0]),
            'learning_rate': individual[1],
            'subsample': np.clip(individual[2], 0.1, 1.0),
            'colsample_bytree': np.clip(individual[3], 0.1, 1.0),
            'reg_alpha': max(0, individual[4]),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42
        }

        tfidf_params = {
            'max_features': int(individual[5]),
            'ngram_range': (1, int(individual[6])),
            'stop_words': 'english'
        }

        try:
            vectorizer = TfidfVectorizer(**tfidf_params)
            X_transformed = vectorizer.fit_transform(text_data)

            model = xgb.XGBClassifier(**xgb_params)
            scores = cross_val_score(model, X_transformed, y, cv=5, scoring='accuracy')
            return (np.mean(scores),)
        except Exception as e:
            return (0.0,)

    # Genetic Algorithm setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Mutation operator
    def mutGaussian(individual, mu=0, sigma=0.1, indpb=0.2):
        for i in range(len(individual)):
            param_name = list(PARAM_BOUNDS.keys())[i]
            if random.random() < indpb:
                if param_name in ['n_estimators', 'max_features', 'ngram_range']:
                    individual[i] = int(np.clip(individual[i] + random.gauss(mu, sigma * 100), *PARAM_BOUNDS[param_name]))
                else:
                    individual[i] = float(np.clip(individual[i] + random.gauss(mu, sigma), *PARAM_BOUNDS[param_name]))
        return individual,

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutGaussian, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual, text_data=text_data, y=y)

    population = toolbox.population(n=number_of_population)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(
        population, 
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=number_of_generations,
        halloffame=hof,
        stats=stats,
        verbose=True
    )

    # Get best individual
    best_ind = hof[0]
    best_params = {
        'n_estimators': int(best_ind[0]),
        'learning_rate': best_ind[1],
        'subsample': best_ind[2],
        'colsample_bytree': best_ind[3],
        'reg_alpha': best_ind[4],
        'tfidf__max_features': int(best_ind[5]),
        'tfidf__ngram_range': (1, int(best_ind[6])),
        'tfidf__stop_words': 'english'
    }

    return best_params



def genetic_algorithm_rf_with_tfidf(X, y, params_bound=None, number_of_generations=10, number_of_population=50):
    if params_bound:
        PARAM_BOUNDS = params_bound
    else:
        PARAM_BOUNDS = {
            'n_estimators': (100, 500),
            'max_depth': (3, 20),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features_tfidf': (100, 1000),     
            'ngram_range': (1, 5),               
        }

    def create_individual():
        return [
            random.randint(*PARAM_BOUNDS['n_estimators']),
            random.randint(*PARAM_BOUNDS['max_depth']),
            random.randint(*PARAM_BOUNDS['min_samples_split']),
            random.randint(*PARAM_BOUNDS['min_samples_leaf']),
            random.randint(*PARAM_BOUNDS['max_features_tfidf']),  # TF-IDF
            random.randint(*PARAM_BOUNDS['ngram_range'])          # TF-IDF
        ]

    def evaluate_individual(individual, X, y):
        rf_params = {
            'n_estimators': individual[0],
            'max_depth': None if individual[1] == 0 else individual[1],
            'min_samples_split': max(2, individual[2]),
            'min_samples_leaf': max(1, individual[3])
        }

        tfidf_params = {
            'max_features': individual[4],
            'ngram_range': (1, individual[5]),
            'stop_words': 'english'
        }

        try:
            vectorizer = TfidfVectorizer(**tfidf_params)
            X_tfidf = vectorizer.fit_transform(X)
            model = RandomForestClassifier(**rf_params)
            scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy')
            return (np.mean(scores),)
        except Exception as e:
            print(f"Error with params {rf_params} and TF-IDF {tfidf_params}: {str(e)}")
            return (0.0,)

    # GA setup
    creator.create("FitnessMax_rf", base.Fitness, weights=(1.0,))
    creator.create("Individual_rf", list, fitness=creator.FitnessMax_rf)

    toolbox_rf = base.Toolbox()
    toolbox_rf.register("individual", tools.initIterate, creator.Individual_rf, create_individual)
    toolbox_rf.register("population", tools.initRepeat, list, toolbox_rf.individual)

    def mutMixed(individual, indpb=0.2):
        for i in range(len(individual)):
            if random.random() < indpb:
                param_name = list(PARAM_BOUNDS.keys())[i]
                new_value = individual[i] + random.randint(-2, 2)
                individual[i] = int(np.clip(new_value, *PARAM_BOUNDS[param_name]))
        return individual,

    toolbox_rf.register("mate", tools.cxTwoPoint)
    toolbox_rf.register("mutate", mutMixed, indpb=0.2)
    toolbox_rf.register("select", tools.selTournament, tournsize=3)
    toolbox_rf.register("evaluate", evaluate_individual, X=X, y=y)

    population = toolbox_rf.population(n=number_of_population)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox_rf,
        cxpb=0.7,
        mutpb=0.2,
        ngen=number_of_generations,
        halloffame=hof,
        stats=stats,
        verbose=True
    )

    # Return best found
    best_ind = hof[0]
    best_params = {
        'rf_params': {
            'n_estimators': best_ind[0],
            'max_depth': None if best_ind[1] == 0 else best_ind[1],
            'min_samples_split': max(2, best_ind[2]),
            'min_samples_leaf': max(1, best_ind[3])
        },
        'tfidf_params': {
            'max_features': best_ind[4],
            'ngram_range': (1, best_ind[5]),
            'stop_words': 'english'
        }
    }
    return best_params
