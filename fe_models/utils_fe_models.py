from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def grid_search_logistic(X, y, params_bound=None, cv=5):
    if params_bound:
        param_grid = params_bound
    else:
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['saga', 'liblinear', 'lbfgs'],
        }

    logreg = LogisticRegression(max_iter=10000)
    
    grid_search = GridSearchCV(logreg, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_params_, grid_search.best_score_


def grid_search_svc(X, y, params_bound=None, cv=5):
    if params_bound:
        param_grid = params_bound
    else:
        param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svc = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_

    return best_model, grid_search.best_params_, grid_search.best_score_


def genetic_algorithm_xgb(X, y, params_bound=None, number_of_generations=10, number_of_population=50):
    if params_bound:
        PARAM_BOUNDS = params_bound
    else:
        PARAM_BOUNDS = {
            'n_estimators': (50, 500),
            'learning_rate': (0.001, 0.3),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0, 10)
        }

    # Create individual
    def create_individual():
        return [
            random.randint(*PARAM_BOUNDS['n_estimators']),
            random.uniform(*PARAM_BOUNDS['learning_rate']),
            random.uniform(*PARAM_BOUNDS['subsample']),
            random.uniform(*PARAM_BOUNDS['colsample_bytree']),
            random.uniform(*PARAM_BOUNDS['reg_alpha'])
        ]

    # Evaluation with cross-validation
    def evaluate_individual(individual, X, y):
        params = {
            'n_estimators': max(1, int(individual[0])),
            'learning_rate': max(0.001, individual[1]),
            'subsample': np.clip(individual[2], 0.1, 1.0),
            'colsample_bytree': np.clip(individual[3], 0.1, 1.0),
            'reg_alpha': max(0, individual[4]),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        try:
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            return (np.mean(scores),)
        except:
            return (0.0,)

    # Genetic Algorithm setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom mutation operator
    def mutGaussian(individual, mu=0, sigma=0.1, indpb=0.2):
        for i in range(len(individual)):
            if random.random() < indpb:
                param_name = list(PARAM_BOUNDS.keys())[i]
                new_value = individual[i] + random.gauss(mu, sigma)
                individual[i] = np.clip(new_value, *PARAM_BOUNDS[param_name])
        return individual,

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutGaussian, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual, X=X, y=y)

    # Run GA
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

    # Get best parameters
    best_ind = hof[0]
    best_params = {
        'n_estimators': max(1, int(best_ind[0])),
        'learning_rate': max(0.001, best_ind[1]),
        'subsample': np.clip(best_ind[2], 0.1, 1.0),
        'colsample_bytree': np.clip(best_ind[3], 0.1, 1.0),
        'reg_alpha': max(0, best_ind[4])
    }
    return best_params




def genetic_algorithm_randomforest(X, y, params_bound=None, number_of_generations=10, number_of_population=50):
    if params_bound:
        PARAM_BOUNDS = params_bound
    else:
        PARAM_BOUNDS = {
            'n_estimators': (100, 500),          # Number of trees
            'max_depth': (3, 20),                # Maximum tree depth (None for unlimited)
            'min_samples_split': (2, 10),        # Min samples to split node
            'min_samples_leaf': (1, 10),         # Min samples at leaf node
        }

    def create_individual():
        return [
            random.randint(*PARAM_BOUNDS['n_estimators']),
            random.randint(*PARAM_BOUNDS['max_depth']),
            random.randint(*PARAM_BOUNDS['min_samples_split']),
            random.randint(*PARAM_BOUNDS['min_samples_leaf'])
        ]

    def evaluate_individual(individual, X, y):
        params = {
            'n_estimators': individual[0],
            'max_depth': None if individual[1] == 0 else individual[1],
            'min_samples_split': max(2, individual[2]),  # Ensure at least 2
            'min_samples_leaf': max(1, individual[3]),   # Ensure at least 1
        }
        
        try:
            model = RandomForestClassifier(**params)
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            return (np.mean(scores),)
        except Exception as e:
            print(f"Error with params {params}: {str(e)}")
            return (0.0,)

    # Genetic Algorithm setup
    creator.create("FitnessMax_rf", base.Fitness, weights=(1.0,))
    creator.create("Individual_rf", list, fitness=creator.FitnessMax)

    toolbox_rf = base.Toolbox()
    toolbox_rf.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox_rf.register("population", tools.initRepeat, list, toolbox_rf.individual)

    def mutMixed(individual, indpb=0.2):
        for i in range(len(individual)):
            if random.random() < indpb:
                param_name = list(PARAM_BOUNDS.keys())[i]
                if param_name in ['n_estimators', 'max_depth', 
                                'min_samples_split', 'min_samples_leaf']:
                    new_value = individual[i] + random.randint(-2, 2)
                    individual[i] = int(np.clip(new_value, *PARAM_BOUNDS[param_name]))
        return individual,


    # toolbox_rf.register("mate", tools.cxBlend, alpha=0.5)
    toolbox_rf.register("mate", tools.cxTwoPoint)  # avoids float crossover entirely
    toolbox_rf.register("mutate", mutMixed, indpb=0.2)  # Removed sigma parameter
    toolbox_rf.register("select", tools.selTournament, tournsize=3)
    toolbox_rf.register("evaluate", evaluate_individual, X=X, y=y)


    # Run GA
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

    # Get best parameters
    best_ind = hof[0]
    best_params = {
        'n_estimators': best_ind[0],
        'max_depth': None if best_ind[1] == 0 else best_ind[1],
        'min_samples_split': max(2, best_ind[2]),
        'min_samples_leaf': max(1, best_ind[3])
    }
    return best_params







