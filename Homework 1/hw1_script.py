import argparse
from importlib import import_module
import os
from pathlib import Path
from types import ModuleType
from typing import Callable, Literal

import numpy as np
from numpy import ndarray
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


def problem_1_part_c(
    hw: ModuleType,
    outname: str | None,
    atol: float,
    seed: int
) -> None:
    '''DO NOT TOUCH! Runs Problem 1 Part C.

    Requires implementing the following functions:
        sigmoid()
        softmax()
        nll_binary()
        nll_multiclass()

    Args:
        outname: name of output file (without extension)
        atol: absolute tolerance for flop numerical errors
        seed: for reproducability
    '''
    print('\n---------- Problem 1 Part C ----------')
    prng = np.random.default_rng(seed=seed)

    n, d = 1000, 8
    X = prng.standard_normal(size=(n, d))
    Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)

    ## Make binary fake data
    w_true = prng.standard_normal(size=(d+1))
    z_true = np.matvec(Xaug, w_true)
    y = np.rint(hw.sigmoid(z_true)).astype(int)

    ## Compute BCE from scratch
    w = prng.standard_normal(size=w_true.shape)
    nllbin_scratch = hw.nll_binary(X, w, y)

    ## Compute BCE from sklearn
    z = np.matvec(Xaug, w)
    ypred = hw.sigmoid(z)
    nllbin_sklearn = log_loss(y, ypred)

    ## Compare BCE scratch and sklearn
    print('Binary NLL from scratch:', nllbin_scratch)
    print('Binary NLL from sklearn:', nllbin_sklearn)
    print('Equal?', nllbin_scratch == nllbin_sklearn)
    print('Almost equal?', np.abs(nllbin_scratch - nllbin_sklearn) < atol)

    ## Make categorical fake data
    k = 4
    encoder = OneHotEncoder(sparse_output=False)
    W_true = prng.standard_normal(size=(d+1, k))
    Z_true = Xaug @ W_true
    Y_true = hw.softmax(Z_true).argmax(axis=1, keepdims=True)
    Y_true_onehot = encoder.fit_transform(Y_true).astype(int)

    ## Compute CCE from scratch
    W = prng.standard_normal(size=W_true.shape)
    nllmult_scratch = hw.nll_multiclass(X, W, Y_true_onehot)

    ## Compute CCE from sklearn
    Z = Xaug @ W
    Ypred = hw.softmax(Z)
    nllmult_sklearn = log_loss(Y_true_onehot, Ypred)

    ## Compare CCE scratch and sklearn
    print('\nCategorical CE from scratch:', nllmult_scratch)
    print('Categorical CE from sklearn:', nllmult_sklearn)
    print('Equal?', nllmult_scratch == nllmult_sklearn)
    print('Almost equal?', np.abs(nllmult_scratch - nllmult_sklearn) < atol)

    # if outname is not None:
        # dir_path = Path(__file__).resolve().parent
        # answers = np.array([nllbin_scratch, nllmult_scratch])
        # print(f'Generating answers file: "{outname}.npy"')
        # np.save(f'{outname}.npy', answers)
    print('--------------------------------------')


def problem_2(
    hw: ModuleType,
    seed: int
) -> None:
    '''DO NOT TOUCH! Runs Problem 2.

    ivm prefix stands for (GD) iterations v MSE
    rvf prefix stands for runtime v num features

    This function will produce the following plots (all png files):
        1) rvf_fullrank
        2) ivm_fullrank_fullscale
        3) ivm_fullrank_zoomed
        4) rvf_singular
        5) ivm_singular_fullscale
        6) ivm_singular_zoomed

    Requires implementing the following functions:
        linreg_ne()
        linreg_gd()
        MSE()
        plot_runtime_v_feature_dim()
        plot_gd_iters_v_mse()
    '''
    print('\n----------    Problem 2     ----------')
    prng = np.random.default_rng(seed=seed)

    ## Fake data params
    n = 1000  ## train set n samples
    ntest = 200  ## test set n samples
    ds = [10, 25, 50, 75, 100, 250, 500, 750, 1000]  ## num input features
    m = 25  ## num prediction features
    sigma = 0.1  ## noise scale for data

    ## NE ridge regularization params
    lmbda = 1.

    ## GD params
    n_iters = 50
    lr = 0.0001

    ## Common arrays to hold results in. Reused for singular X.
    mses_ne = np.zeros(len(ds))
    mses_ne_ridge = np.zeros(len(ds))
    mses_gd = np.zeros((len(ds), n_iters))
    runtimes_ne = np.zeros(len(ds))
    runtimes_ne_ridge = np.zeros(len(ds))
    runtimes_gd = np.zeros(len(ds))

    # Loop over singular and near-singular X
    for singular in [False, True]:
        if singular:
            print('\nNear singular X...')
            rvf_title = 'Runtime v. Feature Dims (Near Singular X)'
            rvf_plotname = 'rvf_singular'
            ivm_title = 'GD Iters v. MSE (Near Singular X)'
            ivm_plotname = 'ivm_singular'
        else:
            print('\nFull rank X...')
            rvf_title = 'Runtime v. Feature Dims (Full Rank X)'
            rvf_plotname = 'rvf_fullrank'
            ivm_title = 'GD Iters v. MSE (Full Rank X)'
            ivm_plotname = 'ivm_fullrank'

        # Loop over feature dimensions
        for i, d in enumerate(ds):
            print('  Num. Features:', d)
            ## Make synthetic train and test data
            X = prng.standard_normal((n, d))
            Xtest = prng.standard_normal((ntest, d))
            if singular:
                r = d // 10  ## set rank = d * 0.1
                U, S, VT = np.linalg.svd(X, full_matrices=False)
                S[-r:] = 1e-10  ## make near singular
                ## reconstruct near singular X and normalize to mean 0, std 1
                X = U @ np.diag(S) @ VT
                X -= X.mean(axis=0, keepdims=True)
                X /= X.std(axis=0, keepdims=True)
                U, S, VT = np.linalg.svd(Xtest, full_matrices=False)
                S[-r:] = 1e-10  ## make near singular
                ## reconstruct near singular Xtest and normalize to mean 0, std 1
                Xtest = U @ np.diag(S) @ VT
                Xtest -= Xtest.mean(axis=0, keepdims=True)
                Xtest /= Xtest.std(axis=0, keepdims=True)
            Xaug = np.concatenate((np.ones((n, 1)), X), axis=1)
            Xtestaug = np.concatenate((np.ones((ntest, 1)), Xtest), axis=1)
            W = prng.standard_normal((d+1, m))  ## make true weights
            eps = prng.standard_normal((n, m)) * sigma  ## eps \sim N(0, sigma^2)
            Y = (Xaug @ W) + eps
            epstest = prng.standard_normal((ntest, m)) * sigma
            Ytest = (Xtestaug @ W) + epstest

            ## Train NE, NE with ridge, and GD on train set
            W_ne, runtime_ne = hw.linreg_ne(X, Y, None)
            W_ne_ridge, runtime_ne_ridge = hw.linreg_ne(X, Y, lmbda)
            Ws_gd, runtime_gd = hw.linreg_gd(X, Y, n_iters, lr)

            ## Predict and evaluate methods on test set
            Ytest_ne = Xtestaug @ W_ne
            Ytest_ne_ridge = Xtestaug @ W_ne_ridge
            Ystest_gd = Xtestaug[None] @ Ws_gd  ## broadcast Xtestaug to all Ws

            mse_ne = hw.MSE(Ytest, Ytest_ne)
            mse_ne_ridge = hw.MSE(Ytest, Ytest_ne_ridge)
            mse_gd_iters = np.array([hw.MSE(Ytest, Ystest_gd[j]) for j in range(n_iters)])

            ## Record everything
            mses_ne[i] = mse_ne
            mses_ne_ridge[i] = mse_ne_ridge
            mses_gd[i] = mse_gd_iters
            runtimes_ne[i] = runtime_ne
            runtimes_ne_ridge[i] = runtime_ne_ridge
            runtimes_gd[i] = runtime_gd

        ## Make plots
        hw.plot_runtime_v_feature_dim(
            ds, runtimes_ne, runtimes_gd,
            rvf_title, rvf_plotname
        )
        hw.plot_gd_iters_v_mse(
            ds, mses_ne, mses_ne_ridge, mses_gd,
            ivm_title, ivm_plotname
        )
    print('--------------------------------------')


def problem_3_skeleton(
    hw: ModuleType,
    thresh: float,
    lossname: Literal['two_hole'] | Literal['multi_modal'],
    lossfn: Callable[[ndarray], float],
    gradfn: Callable[[ndarray], tuple[ndarray, ndarray]],
    seed: int
) -> None:
    '''DO NOT TOUCH! Skeleton for Problem 3 two hole and multi modal loss.'''
    prng = np.random.default_rng(seed=seed)

    ## Fixed hyperparams
    n_trials = 20
    max_iters = 300
    noise_decay = 0.995
    escape_chance = 0.25
    atol = 1e-6

    ## Grid search hyperparams
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    batch_sizes = [1, 4, 16, 64]  # Simulate different batch sizes
    noise_scales = [0.1, 0.5, 1.0, 2.0]

    ## Same starting point as in linked notebook
    start_point = np.array([0.9, 0.9])
    print(f"Starting point: ({start_point[0]:.1f}, {start_point[1]:.1f})")

    losses = np.zeros((len(learning_rates), len(batch_sizes), len(noise_scales), n_trials))
    runtimes = np.zeros(losses.shape)
    conv_iters = np.zeros(losses.shape)

    pbar = tqdm(
        range(np.prod(losses.shape)),
        desc=f'Hyperparam Sweep Progress ({lossname})'
    )
    for i, lr in enumerate(learning_rates):
        for j, b in enumerate(batch_sizes):
            for k, sigma in enumerate(noise_scales):
                for ell in range(n_trials):
                    w, runtime, iteration = hw.run_sgd_improved_analysis(
                        start_point,
                        gradfn,
                        lr,
                        max_iters,
                        sigma,
                        b,
                        noise_decay,
                        escape_chance,
                        atol,
                        prng
                    )
                    losses[i, j, k, ell] = lossfn(w)
                    runtimes[i, j, k, ell] = runtime
                    conv_iters[i, j, k, ell] = iteration
                    pbar.update(1)
    pbar.close()
    print('Hyperparam Sweep Done')
    escaped = hw.check_escaped(losses, thresh)

    hw.plot_heatmaps(
        lossname,
        learning_rates,
        batch_sizes,
        noise_scales,
        losses,
        runtimes,
        conv_iters,
        escaped
    )


def problem_3_part_b(
    hw: ModuleType,
    seed: int
) -> None:
    '''DO NOT TOUCH! Runs Problem 3 part b.

    Requires implementing the following functions:
        linreg_ne()
        linreg_gd()
        MSE()
        plot_runtime_v_feature_dim()
        plot_gd_iters_v_mse()

    Generates the following plot:
        two_hole_heatmaps
    '''
    print('\n---------- Problem 3 Part B ----------')
    two_hole_thresh = -3.0
    problem_3_skeleton(
        hw,
        two_hole_thresh,
        'two_hole',
        hw.loss_function,
        hw.get_gradient_components,
        seed
    )
    print('\n--------------------------------------')


def problem_3_part_c(
    hw: ModuleType,
    seed: int
) -> None:
    '''DO NOT TOUCH! Runs Problem 3 part c.

    Requires implementing the additional functions:
        multi_modal_loss()
        multi_modal_grad_components()

    Generates the following plot:
        multi_modal_heatmaps
    '''
    print('\n---------- Problem 3 Part C ----------')
    multi_modal_thresh = -3.0
    problem_3_skeleton(
        hw,
        multi_modal_thresh,
        'multi_modal',
        hw.multi_modal_loss,
        hw.multi_modal_grad_components,
        seed
    )
    print('\n--------------------------------------')


def problem_4(
    hw: ModuleType,
    seed: int
) -> None:
    '''DO NOT TOUCH! Runs Problem 4.

    Requires implementing the additional functions/class:
        SimplePerceptron
        create_nonlinear_features()

    Generates the following plots:
        xor_plot
        decision_boundary
    '''
    print('\n----------    Problem 4     ----------')
    prng = np.random.default_rng(seed=seed)

    X, y = hw.create_xor_dataset()
    print('XOR Dataset - The Classic Non-Linearly Separable Problem:')
    print('Inputs (X):')
    print(X)
    print('Outputs (y):')
    print(y)
    print('\nNotice: Points (0, 1) and (1, 0) have output 1, while (0, 0) and (1, 1) have output 0')
    print('No single straight line can separate these two classes!')

    hw.plot_xor_data(X, y)

    learning_rate = 0.1
    max_epochs = 300
    perceptron = hw.SimplePerceptron(learning_rate, max_epochs, prng)
    perceptron.fit(X, y)

    # Make predictions and evaluate
    predictions = perceptron.predict(X)
    accuracy = accuracy_score(y, predictions)
    print('\nFinal Results:')
    print(f'Accuracy: {accuracy:.2f} (Perfect would be 1.00)')
    print(f'Final weights: {perceptron.weights}')
    print(f'Final bias: {perceptron.bias:.3f}')

    print('\nPredictions vs True values:')
    print('Input  | True | Predicted | Correct?')
    print('-' * 36)
    for i in range(len(X)):
        correct = "o" if predictions[i] == y[i] else "x"
        print(f'{X[i]}  |  {y[i]}   |     {predictions[i]}     |    {correct}')

    # Visualize decision boundary
    hw.visualize_decision_boundary(X, y, perceptron)

    # Enhance XOR dataset with product feature
    X_enhanced = hw.create_nonlinear_features(X)
    print('\nOriginal XOR problem:')
    print('Inputs (x1, x2):')
    print(X)
    print('\nEnhanced with non-linear features:')
    print('Inputs (x1, x2, x1 * x2):')
    print(X_enhanced)
    print('\nOutputs:', y)

    print('\nKey insight: In the enhanced space, the problem becomes linearly separable!')
    print('Notice how the x1 * x2 feature helps distinguish the classes:')
    for i in range(len(X_enhanced)):
        print(f'Point {X_enhanced[i]} --> class {y[i]}')

    # Train a perceptron on the enhanced features
    print('\nTraining perceptron on enhanced features...')
    enhanced_perceptron = Perceptron(random_state=seed, max_iter=max_epochs)
    enhanced_perceptron.fit(X_enhanced, y)

    enhanced_predictions = enhanced_perceptron.predict(X_enhanced)
    enhanced_accuracy = accuracy_score(y, enhanced_predictions)

    print('\nResults with non-linear features:')
    print(f'Accuracy: {enhanced_accuracy:.2f} (Perfect!)')
    print(f'Weights: {enhanced_perceptron.coef_[0]}')
    print(f'Bias: {enhanced_perceptron.intercept_[0]:.3f}')

    print('\nPredictions vs True values:')
    print('Enhanced Input | True | Predicted | Correct?')
    print('-' * 44)
    for i in range(len(X_enhanced)):
        correct = 'o' if enhanced_predictions[i] == y[i] else 'x'
        print(f'{X_enhanced[i]}        |  {y[i]}   |     {enhanced_predictions[i]}     |    {correct}')

    print('\nConclusion: By adding non-linear features, we made XOR linearly separable!')
    print('This is exactly what hidden layers in neural networks do automatically!')
    print('--------------------------------------')


def main() -> None:
    '''Runs program and outputs solution file.

    You may comment out individual problems during debugging
    but instructors will run all problem parts.

    When all parts are run the following plots (png) will be
    generated in the current directory:
        1) rvf_fullrank
        2) ivm_fullrank_fullscale
        3) ivm_fullrank_zoomed
        4) rvf_singular
        5) ivm_singular_fullscale
        6) ivm_singular_zoomed
        7) two_hole_heatmaps
        8) multi_modal_heatmaps
        9) xor_plot
        10) decision_boundary
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw', type=str, default='')
    parser.add_argument('--outname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    hw = import_module(os.path.join(args.hw, 'hw1_impl'))
    outname = args.outname
    seed = args.seed
    print('Using seed:', seed)

    atol = 1e-10  ## catch slight numerical errors from implementations

    problem_1_part_c(hw, outname, atol, seed)
    problem_2(hw, seed)
    problem_3_part_b(hw, seed)
    problem_3_part_c(hw, seed)
    problem_4(hw, seed)


if __name__ == '__main__':
    main()
