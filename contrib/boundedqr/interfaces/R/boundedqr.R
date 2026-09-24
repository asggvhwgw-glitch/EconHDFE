# BoundedQR statistical-software bridge. GPL-3.0-or-later, as the project.
# Existing quantreg conventions and CFM preprocessing are credited in README.md.
# This file requires only quantreg/jsonlite; no reticulate or Rcpp dependency.

boundedqr_cluster <- function(X, y, cluster, tau = .5, reps = 199L,
                              seed = 20260915L, backend = "cpu", batch_size = 16L,
                              threads = 8L, cpu_factor = 1, audit_backend = "cpu",
                              verify_radius = TRUE, level = .95,
                              multipliers = NULL, python = NULL,
                              project_root = getOption("boundedqr.project_root", getwd()),
                              temp_dir = NULL, keep_files = FALSE,
                              positive_pseudo = FALSE) {
    wall <- proc.time()[["elapsed"]]
    fail <- function(...) stop(..., call. = FALSE)
    for (package in c("quantreg", "jsonlite")) {
        if (!requireNamespace(package, quietly = TRUE))
            fail("Install the R package '", package, "' before calling boundedqr_cluster().")
    }
    if (!is.matrix(X) || !is.numeric(X) || !length(X))
        fail("X must be a nonempty numeric dense matrix; use model.matrix() explicitly.")
    storage.mode(X) <- "double"
    y <- as.double(y)
    n <- nrow(X); p <- ncol(X)
    if (n < p || length(y) != n || any(!is.finite(X)) || any(!is.finite(y)))
        fail("Require finite X/y, n >= p and length(y) = n. Handle missing observations explicitly.")
    if (length(cluster) != n || anyNA(cluster) || is.list(cluster))
        fail("cluster must be a nonmissing atomic vector of length n.")
    tau <- as.double(tau)
    if (!length(tau) || any(!is.finite(tau)) || any(tau <= 0 | tau >= 1))
        fail("tau must contain quantiles strictly inside (0,1).")
    integer_option <- function(value, name, lower = 1L) {
        if (length(value) != 1L || !is.numeric(value) || !is.finite(value) ||
            value < lower || value != floor(value) || value > .Machine$integer.max)
            fail(name, " must be an integer >= ", lower, ".")
        as.integer(value)
    }
    reps <- integer_option(reps, "reps", 2L)
    batch_size <- integer_option(batch_size, "batch_size")
    threads <- integer_option(threads, "threads")
    seed <- integer_option(seed, "seed", 0L)
    if (length(backend) != 1L || !backend %in% c("cpu", "gpu"))
        fail("backend must be 'cpu' or 'gpu'.")
    if (length(audit_backend) != 1L || !audit_backend %in% c("cpu", "gpu", "gpu_fused"))
        fail("audit_backend must be 'cpu', 'gpu', or 'gpu_fused'.")
    if (!is.logical(positive_pseudo) || length(positive_pseudo) != 1L || is.na(positive_pseudo))
        fail("positive_pseudo must be TRUE or FALSE.")
    if (length(cpu_factor) != 1L || !is.finite(cpu_factor) || cpu_factor <= 0 ||
        length(level) != 1L || !is.finite(level) || level <= 0 || level >= 1)
        fail("Require cpu_factor > 0 and 0 < level < 1.")
    if (!is.logical(verify_radius) || length(verify_radius) != 1L || is.na(verify_radius) ||
        !is.logical(keep_files) || length(keep_files) != 1L || is.na(keep_files))
        fail("verify_radius and keep_files must be TRUE or FALSE.")
    root <- normalizePath(project_root, winslash = "/", mustWork = TRUE)
    source_dir <- file.path(root, "src")
    if (!file.exists(file.path(source_dir, "boundedqr", "cli.py")))
        fail("project_root must be the BoundedQR project containing src/boundedqr/cli.py.")
    if (is.null(python)) {
        local_python <- file.path(root, ".venv", if (.Platform$OS.type == "windows") "Scripts/python.exe" else "bin/python")
        python <- if (file.exists(local_python)) local_python else Sys.which("python3")
    }
    if (length(python) != 1L || !nzchar(python) || !file.exists(python))
        fail("Specify python as the existing Python executable with BoundedQR dependencies installed.")
    python <- normalizePath(python, winslash = "/", mustWork = TRUE)
    if (is.null(temp_dir)) temp_dir <- file.path(root, ".cache", "r_bridge")
    compare_path <- function(value) if (.Platform$OS.type == "windows") tolower(value) else value
    # Validate before creating directories. Reject ambiguous .. components;
    # resolve the existing ancestor to detect paths outside this project.
    components <- strsplit(gsub("\\\\", "/", path.expand(temp_dir)), "/", fixed = TRUE)[[1]]
    if (any(components == "..")) fail("temp_dir must use a direct path without '..' components.")
    ancestor <- temp_dir
    while (!dir.exists(ancestor)) {
        parent <- dirname(ancestor)
        if (identical(parent, ancestor)) fail("Cannot resolve temp_dir ancestor.")
        ancestor <- parent
    }
    ancestor <- normalizePath(ancestor, winslash = "/", mustWork = TRUE)
    if (!startsWith(compare_path(paste0(ancestor, "/")), compare_path(paste0(root, "/"))))
        fail("temp_dir must remain inside project_root.")
    dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
    temp_root <- normalizePath(temp_dir, winslash = "/", mustWork = TRUE)
    if (!startsWith(compare_path(paste0(temp_root, "/")), compare_path(paste0(root, "/"))))
        fail("temp_dir must remain inside project_root.")
    job_dir <- tempfile("boundedqr_", tmpdir = temp_root)
    if (!dir.create(job_dir)) fail("Could not create bridge work directory: ", job_dir)
    job_dir <- normalizePath(job_dir, winslash = "/", mustWork = TRUE)
    success <- FALSE
    on.exit(if (success && !keep_files) unlink(job_dir, recursive = TRUE), add = TRUE)
    # shortPathName is important for the bundled Windows R/Python setup under
    # a project path containing Chinese characters; no source files are moved.
    process_path <- function(path) {
        if (.Platform$OS.type == "windows") shortPathName(path) else path
    }
    binary_write_seconds <- 0
    write_array <- function(name, value, shape = dim(value)) {
        tick <- proc.time()[["elapsed"]]
        con <- file(file.path(job_dir, name), "wb")
        on.exit({ close(con); binary_write_seconds <<- binary_write_seconds +
                    proc.time()[["elapsed"]] - tick })
        writeBin(as.double(value), con, size = 8L, endian = "little")
        list(file = name, shape = I(as.integer(shape)))
    }
    labels <- unique(cluster) # exactly quantreg::boot.rq's first-occurrence order
    codes <- match(cluster, labels)
    groups <- length(labels)
    if (groups < 2L) fail("At least two clusters are required.")
    if (!is.null(multipliers)) {
        if (!is.matrix(multipliers) || !is.numeric(multipliers) ||
            !identical(dim(multipliers), c(groups, reps)) || any(!is.finite(multipliers)))
            fail("multipliers must be a finite G-by-reps matrix in unique(cluster) order.")
    }
    x_spec <- write_array("x.bin", X, c(n, p))
    y_spec <- write_array("y.bin", y, n)
    q <- length(tau)
    base <- matrix(NA_real_, q, p)
    scores <- vector("list", q)
    jobs <- vector("list", q)
    baseline_warnings <- vector("list", q)
    baseline_seconds <- numeric(q)
    score_aggregation_seconds <- numeric(q)
    for (j in seq_len(q)) {
        tick <- proc.time()[["elapsed"]]
        caught <- character()
        fit <- withCallingHandlers(quantreg::rq.fit.fnb(X, y, tau[j]), warning = function(w) {
            caught <<- c(caught, conditionMessage(w))
        })
        baseline_seconds[j] <- proc.time()[["elapsed"]] - tick
        baseline_warnings[[j]] <- caught
        base[j, ] <- as.double(fit$coefficients)
        residual <- as.double(fit$resid)
        if (length(residual) != n || any(!is.finite(residual)) || any(!is.finite(base[j, ])))
            fail("quantreg returned a nonfinite baseline at tau=", tau[j], "; bridge files: ", job_dir)
        # DO NOT recompute y-X%*%beta, round zero residuals, or use a KKT dual.
        score_tick <- proc.time()[["elapsed"]]
        psi <- (residual < 0) - tau[j]
        scores[[j]] <- rowsum(X * psi, codes, reorder = TRUE)
        score_aggregation_seconds[j] <- proc.time()[["elapsed"]] - score_tick
        jobs[[j]] <- list(tau = tau[j],
                          beta0 = write_array(paste0("beta0_", j, ".bin"), base[j, ], p),
                          W = list(file = paste0("W_", j, ".bin"), shape = I(c(p, reps))))
    }
    # Preserve the caller's RNG state. A single column-major sample stream is
    # shared across quantiles, so changing batch_size leaves multipliers fixed.
    if (is.null(multipliers)) {
        had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
        if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
        on.exit(if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
                else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
                    rm(".Random.seed", envir = .GlobalEnv), add = TRUE)
        set.seed(seed)
    }
    h <- c(-(sqrt(5) - 1) / 2, (sqrt(5) + 1) / 2)
    hp <- c((sqrt(5) + 1) / sqrt(20), (sqrt(5) - 1) / sqrt(20))
    connections <- lapply(jobs, function(job) file(file.path(job_dir, job$W$file), "wb"))
    on.exit(for (con in connections) try(close(con), silent = TRUE), add = TRUE)
    gradient_tick <- proc.time()[["elapsed"]]
    for (first in seq.int(1L, reps, by = batch_size)) {
        at <- seq.int(first, min(reps, first + batch_size - 1L))
        v <- if (is.null(multipliers))
            matrix(sample(h, groups * length(at), prob = hp, replace = TRUE), groups, length(at))
            else multipliers[, at, drop = FALSE]
        for (j in seq_len(q)) {
            w <- t(scores[[j]]) %*% v
            write_tick <- proc.time()[["elapsed"]]
            writeBin(as.double(w), connections[[j]], size = 8L, endian = "little")
            binary_write_seconds <- binary_write_seconds + proc.time()[["elapsed"]] - write_tick
        }
    }
    write_tick <- proc.time()[["elapsed"]]
    for (con in connections) close(con)
    binary_write_seconds <- binary_write_seconds + proc.time()[["elapsed"]] - write_tick
    connections <- list()
    gradient_seconds <- proc.time()[["elapsed"]] - gradient_tick
    rm(scores, psi, residual, v, w)
    request <- list(protocol_version = 1L, dtype = "float64", endian = "little", order = "F",
                    n = n, p = p, reps = reps, x = x_spec, y = y_spec, jobs = jobs, level = level,
                    solver = list(backend = backend, batch_size = batch_size, threads = threads,
                                  cpu_factor = cpu_factor, audit_backend = audit_backend,
                                  verify_radius = verify_radius, repair = "batched",
                                  positive_pseudo = positive_pseudo))
    request_path <- file.path(job_dir, "request.json")
    response_path <- file.path(job_dir, "response.json")
    jsonlite::write_json(request, request_path, auto_unbox = TRUE, pretty = TRUE, digits = NA)
    preparation_seconds <- proc.time()[["elapsed"]] - wall
    old_pythonpath <- Sys.getenv("PYTHONPATH", unset = NA_character_)
    on.exit(if (is.na(old_pythonpath)) Sys.unsetenv("PYTHONPATH") else
                Sys.setenv(PYTHONPATH = old_pythonpath), add = TRUE)
    inherited <- if (is.na(old_pythonpath) || !nzchar(old_pythonpath)) character() else old_pythonpath
    Sys.setenv(PYTHONPATH = paste(c(process_path(source_dir), inherited), collapse = .Platform$path.sep))
    stdout <- file.path(job_dir, "python_stdout.log")
    stderr <- file.path(job_dir, "python_stderr.log")
    tick <- proc.time()[["elapsed"]]
    status <- system2(process_path(python),
                      c("-m", "boundedqr", "--request", shQuote(process_path(request_path)),
                        "--response", shQuote(file.path(process_path(job_dir), "response.json"))),
                      stdout = stdout, stderr = stderr, wait = TRUE)
    process_seconds <- proc.time()[["elapsed"]] - tick
    response <- if (file.exists(response_path)) jsonlite::read_json(response_path, simplifyVector = FALSE) else NULL
    if (!identical(as.integer(status), 0L) || is.null(response) || !isTRUE(response$ok)) {
        detail <- if (!is.null(response$message)) response$message else "Python produced no successful response."
        fail("BoundedQR Python bridge failed: ", detail, "\nWork files and stderr retained in: ", job_dir)
    }
    read_array <- function(spec) {
        shape <- as.integer(unlist(spec$shape))
        con <- file(file.path(job_dir, spec$file), "rb")
        on.exit(close(con))
        expected <- prod(shape)
        if (file.info(file.path(job_dir, spec$file))$size != 8 * expected)
            fail("Wrong Python output size: ", spec$file)
        value <- readBin(con, what = double(), n = expected, size = 8L, endian = "little")
        if (length(shape) > 1L) array(value, shape) else value
    }
    output_tick <- proc.time()[["elapsed"]]
    draws <- array(NA_real_, c(reps, q, p))
    radii <- matrix(Inf, reps, q)
    for (j in seq_len(q)) {
        draws[, j, ] <- read_array(response$jobs[[j]]$coefficients)
        radii[, j] <- read_array(response$jobs[[j]]$radii)
    }
    inference <- response$inference
    for (name in names(response$inference_arrays))
        inference[[name]] <- read_array(response$inference_arrays[[name]])
    inference$zero_standard_error <- inference$zero_standard_error != 0
    coefficient_names <- colnames(X)
    if (is.null(coefficient_names)) coefficient_names <- paste0("x", seq_len(p))
    dimnames(base) <- list(as.character(tau), coefficient_names)
    dimnames(draws) <- list(NULL, as.character(tau), coefficient_names)
    colnames(radii) <- as.character(tau)
    output_seconds <- proc.time()[["elapsed"]] - output_tick
    output <- list(quantiles = tau, coefficients = if (q == 1L) base[1, ] else base,
                   bootstrap_coefficients = if (q == 1L) matrix(draws[, 1, ], reps, p,
                                                               dimnames = list(NULL, coefficient_names)) else draws,
                   coefficient_radii = radii, inference = inference,
                   standard_errors = inference$standard_errors,
                   basic_intervals = inference$basic_intervals, uniform_band = inference$uniform_band,
                   cluster_labels = labels,
                   diagnostics = list(python = response$diagnostics, python_timing = response$timing,
                                      timing = list(end_to_end_seconds = proc.time()[["elapsed"]] - wall,
                                                    r_preparation_and_input_seconds = preparation_seconds,
                                                    quantreg_baseline_seconds = baseline_seconds,
                                                    score_aggregation_seconds = score_aggregation_seconds,
                                                    binary_input_write_seconds = binary_write_seconds,
                                                    gradient_and_write_seconds = gradient_seconds,
                                                    python_process_seconds = process_seconds,
                                                    r_output_read_seconds = output_seconds),
                                      quantreg_version = as.character(utils::packageVersion("quantreg")),
                                      baseline_warnings = baseline_warnings,
                                      score_source = "quantreg::rq.fit.fnb original resid; (resid < 0) - tau",
                                      multiplier_source = if (is.null(multipliers)) "shared R Mammen sample stream" else "supplied G-by-reps",
                                      multiplier_order = "unique(cluster), first occurrence",
                                      seed = seed, rng_kind = RNGkind(),
                                      peak_generated_multiplier_shape = c(groups, min(reps, batch_size)),
                                      no_internal_n_by_reps_array = TRUE,
                                      numerical_scope = response$numerical_scope,
                                      nonfinite_json = response$nonfinite_json,
                                      work_dir = if (keep_files) job_dir else NULL))
    class(output) <- "boundedqr_cluster"
    if (verify_radius && any(!is.finite(radii))) {
        warning(sprintf(paste0(
            "BoundedQR: %d of %d draw-quantile coefficient radii are non-finite. ",
            "All draws and point summaries are retained, but the affected point summaries ",
            "have no finite coefficient-error enclosure from this run. ",
            "Non-finite radii do not establish incorrect fits or nonuniqueness. ",
            "Inspect coefficient_radii and diagnostics."),
            sum(!is.finite(radii)), length(radii)), call. = FALSE)
    }
    success <- TRUE
    output
}
