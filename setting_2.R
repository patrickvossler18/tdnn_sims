library(tidyverse)
library(tdnn)
library(kknn)
library(boot)
library(argparser)


# Simulation Parameters ---------------------------------------------------
parser <- arg_parser("Setting 2 params")
parser <-
    add_argument(
        parser,
        "--dimension",
        type = "integer",
        default = 10,
        help = "Dimension for the setting [default 10]"
    )

opt <- parse_args(parser)
p = opt$dimension

data_type = "unif"
draw_random_data <- "fixed"
use_R <- FALSE
est_variance <- TRUE
verbose <- TRUE

num_threads <- min(RcppParallel::defaultNumThreads(), 10)


seed_val <- 1234
num_reps <- 1000
n = 1000
n_test <- 100
c_seq = c(2, 4, 6, 8, 10, 15, 20, 25, 30)
c_seq_fixed = 2
s_1_seq <- seq(1, 100, 1)
dnn_s_seq <- seq(10, 100, 1)
alpha = 0.001




if (data_type == "unif") {
    fixed_test_vector = c(c(0.2, 0.4, 0.6), rep(0.5, p - 3))
} else if (data_type == "normal") {
    fixed_test_vector = c(c(0.5, -0.5, 0.5), rep(0, p - 3))
}

dgp_function = function(x) {
    term1 <- 4 * (4 * x[1] - 2 + 8 * x[2] - 8 * x[2] ^ 2) ^ 2
    term2 <- (3 - 4 * x[2]) ^ 2
    term3 <- 16 * sqrt(x[3] + 1) * (2 * x[3] - 1) ^ 2
    term1 + term2 + term3
}


# Simulation helper functions ---------------------------------------------


make_results_df <- function(truth, predictions, variance, method) {
    result_df <- data.frame(cbind(truth, predictions, variance)) %>%
        mutate(method = method)
    colnames(result_df) <-
        c("truth", "predictions", "variance", "method")
    result_df
}

knn_reg <- function(X,
                    Y,
                    X_test,
                    k_val = 20,
                    est_variance = T) {
    knn_mu <- FNN::knn.reg(X, X_test, Y, k = k_val)$pred
    if (est_variance) {
        knn_mu2 <- FNN::knn.reg(X, X_test, Y ^ 2, k = k_val)$pred
        knn_var <- (knn_mu2 - knn_mu ^ 2) / (k_val - 1)
    } else{
        knn_var <- NA
    }
    list(pred_knn = knn_mu,
         variance_knn = knn_var)
}

make_tuned_knn_results <-
    function(X, Y, X_test, kmax, estimate_variance = T) {
        training_data <- cbind(data.frame(X), data.frame(Y = Y))
        testing_data <- data.frame(X_test)
        
        tuned_knn <- train.kknn(Y ~ ., data = training_data, kmax = 100)
        tuned_k <-  tuned_knn$best.parameters$k
        tuned_results <-
            knn_reg(X, as.numeric(Y), X_test, k_val = tuned_k, estimate_variance)
        tuned_results[["best_k"]] = tuned_k
        return(tuned_results)
    }


# Draw random test data ---------------------------------------------------
set.seed(seed_val)
# fix X_test_random
if (data_type == "unif") {
    X_test_random <- matrix(runif(n_test * p, 0, 1), n_test, p)
} else if (data_type == "normal") {
    X_test_random <- matrix(rnorm(n_test * p), n_test, p)
}

mu_rand <- apply(X_test_random, MARGIN = 1, dgp_function)
mu_fixed <- dgp_function(fixed_test_vector)

# Main simulation loop ----------------------------------------------------
run_sim <-
    function(i,
             n,
             c_seq,
             X_test_random,
             mu_rand,
             mu_fixed,
             draw_random_data,
             est_variance = F,
             verbose = F) {
        print(glue::glue("rep {i}/{num_reps}, data type: {data_type}"))
        if (data_type == "unif") {
            X = matrix(runif(n * p, 0, 1), n, p)
        } else if (data_type == "normal") {
            X = matrix(rnorm(n * p), n, p)
        }
        X_test_fixed <- matrix(fixed_test_vector, 1, p)
        
        epsi = matrix(rnorm(n) , n, 1)
        Y = apply(X, MARGIN = 1, dgp_function) + epsi
        
        if (p > 3) {
            W_0 <- tdnn:::feature_screen(X, Y, alpha = alpha)
        } else{
            W_0 = rep(1, p)
        }
        if (verbose)
            print(W_0)
        
        # TDNN Estimates ----------------------------------------------------------
        if (verbose)
            tictoc::tic("tdnn rand results")
        tdnn_rand_estimate <-
            tdnn:::tune_de_dnn_no_dist_vary_c_cpp_thread(
                X,
                Y,
                X_test_random,
                W0_ = W_0,
                c = c_seq,
                estimate_variance = est_variance,
                bootstrap_iter = 500,
                B_NN = 20,
                num_threads = num_threads
            )
        if (verbose)
            tictoc::toc()
        
        if (verbose)
            tictoc::tic("tdnn fixed results")
        tdnn_fixed_estimate <-
            tdnn:::tune_de_dnn_no_dist_vary_c_cpp_thread(
                X,
                Y,
                X_test_fixed,
                W0_ = W_0,
                c = c_seq_fixed,
                estimate_variance = est_variance,
                bootstrap_iter = 500,
                B_NN = 20
            )
        if (verbose)
            tictoc::toc()
        
        
        tdnn_rand_results <-
            data.frame(
                estimate = tdnn_rand_estimate$estimate_loo,
                loss = (tdnn_rand_estimate$estimate_loo - mu_rand) ^
                    2,
                bias = (tdnn_rand_estimate$estimate_loo - mu_rand),
                s_1 = tdnn_rand_estimate$s_1_B_NN,
                c = tdnn_rand_estimate$c_B_NN,
                variance = if (est_variance) {
                    as.numeric(tdnn_rand_estimate$variance)
                } else{
                    NA
                },
                method = 'tdnn_rand'
            )
        tdnn_fixed_results <-
            data.frame(
                estimate = tdnn_fixed_estimate$estimate_loo,
                loss = (tdnn_fixed_estimate$estimate_loo - mu_fixed) ^
                    2,
                bias = (tdnn_fixed_estimate$estimate_loo - mu_fixed),
                s_1 = tdnn_fixed_estimate$s_1_B_NN,
                c = tdnn_fixed_estimate$c_B_NN,
                variance = if (est_variance) {
                    as.numeric(tdnn_fixed_estimate$variance)
                } else{
                    NA
                },
                method = 'tdnn_fixed'
            )
        
        
        # DNN Estimates ----------------------------------------------------------
        if (verbose)
            tictoc::tic("dnn results")
        dnn_fixed_estimate <-
            tdnn:::tune_dnn_no_dist_thread(
                X,
                Y,
                X_test_fixed,
                s_seq = dnn_s_seq,
                W0_ = W_0,
                estimate_variance = est_variance,
                num_threads = num_threads
            )
        dnn_rand_estimate <-
            tdnn:::tune_dnn_no_dist_thread(
                X,
                Y,
                X_test_random,
                s_seq = dnn_s_seq,
                W0_ = W_0,
                estimate_variance =
                    T,
                num_threads = num_threads
            )
        if (verbose)
            tictoc::toc()
        
        dnn_rand_results <-
            data.frame(
                estimate = dnn_rand_estimate$estimate_loo,
                loss = (dnn_rand_estimate$estimate_loo - mu_rand) ^
                    2,
                bias = (dnn_rand_estimate$estimate_loo - mu_rand),
                s_1 = dnn_rand_estimate$s_1_B_NN,
                variance = if (est_variance) {
                    as.numeric(dnn_rand_estimate$variance)
                } else{
                    NA
                },
                method = 'dnn_rand'
            )
        dnn_fixed_results <-
            data.frame(
                estimate = dnn_fixed_estimate$estimate_loo,
                loss = (dnn_fixed_estimate$estimate_loo - mu_fixed) ^
                    2,
                bias = (dnn_fixed_estimate$estimate_loo - mu_fixed),
                s_1 = dnn_fixed_estimate$s_1_B_NN,
                variance = if (est_variance) {
                    as.numeric(dnn_fixed_estimate$variance)
                } else{
                    NA
                },
                method = 'dnn_fixed'
            )
        # KNN Estimates ----------------------------------------------------------
        if (verbose)
            tictoc::tic("knn results")
        X_filtered <- X[, as.logical(W_0)]
        X_test_random_filtered <- X_test_random[, as.logical(W_0)]
        X_test_fixed_filtered <- X_test_fixed[, as.logical(W_0)]
        knn_rand_estimate <-
            make_tuned_knn_results(
                X_filtered,
                Y,
                X_test_random_filtered,
                kmax = 200,
                estimate_variance = est_variance
            )
        knn_fixed_estimate <-
            make_tuned_knn_results(
                X_filtered,
                Y,
                X_test_fixed_filtered,
                kmax = 200,
                estimate_variance = est_variance
            )
        if (verbose)
            tictoc::toc()
        
        knn_rand_results <-
            data.frame(
                estimate = knn_rand_estimate$pred_knn,
                loss = (knn_rand_estimate$pred_knn - mu_rand) ^
                    2,
                bias = (knn_rand_estimate$pred_knn - mu_rand),
                k = knn_rand_estimate$best_k,
                variance = if (est_variance) {
                    as.numeric(knn_rand_estimate$variance)
                } else{
                    NA
                },
                method = 'knn_rand'
            )
        knn_fixed_results <-
            data.frame(
                estimate = knn_fixed_estimate$pred_knn,
                loss = (knn_fixed_estimate$pred_knn - mu_fixed) ^
                    2,
                bias = (knn_fixed_estimate$pred_knn - mu_fixed),
                k = knn_fixed_estimate$best_k,
                variance = if (est_variance) {
                    as.numeric(knn_fixed_estimate$variance)
                } else{
                    NA
                },
                method = 'knn_fixed'
            )
        # Combine results ----------------------------------------------------
        results <-
            bind_rows(
                tdnn_rand_results,
                tdnn_fixed_results,
                dnn_rand_results,
                dnn_fixed_results,
                knn_rand_results,
                knn_fixed_results
            )
        
        return(results)
    }


set.seed(seed_val)
results <- map_dfr(1:num_reps, function(i) {
    run_sim(
        i,
        n,
        c_seq,
        X_test_random,
        mu_rand,
        mu_fixed,
        draw_random_data,
        est_variance = est_variance,
        verbose = verbose
    )
})


# View and Save Results ---------------------------------------------------
results_data <- list(
    results_df = results,
    c_seq = c_seq,
    c_seq_fixed = c_seq_fixed,
    mu_rand = mu_rand,
    mu_fixed = mu_fixed,
    p = p,
    n = n,
    num_reps = num_reps,
    n_test = n_test,
    data_type = data_type,
    fixed_test_vector = fixed_test_vector,
    draw_random_data = draw_random_data,
    dgp_function = dgp_function
)
file_path <-
    glue::glue(
        "setting_2_p_{p}_{data_type}_data_{draw_random_data}_tuned_de_dnn_{str_replace_all(strftime(Sys.time(), '%Y-%m-%d_%H_%M'),'-', '_')}.rds"
    )
results_data %>%
    write_rds(., file_path)

