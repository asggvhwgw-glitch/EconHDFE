# Synthetic R-to-Python bridge and independent quantreg replay check.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(if(length(args)) args[1] else ".", winslash="/", mustWork=TRUE)
python <- if(length(args)>1) args[2] else Sys.which("python")
source(file.path(root, "interfaces", "R", "boundedqr.R"))
set.seed(123)
n <- 240L
g <- rep(seq_len(40), each=6)
X <- cbind(1, matrix(rnorm(n*2), n, 2))
y <- as.vector(X %*% c(1,.7,-.3) + .3*rnorm(40)[g] + rnorm(n))
h <- c(-(sqrt(5)-1)/2, (sqrt(5)+1)/2)
hp <- c((sqrt(5)+1)/sqrt(20), (sqrt(5)-1)/sqrt(20))
V <- matrix(sample(h, 40*9, replace=TRUE, prob=hp), 40, 9)
out <- boundedqr_cluster(X, y, g, tau=.5, reps=9, multipliers=V,
                         project_root=root, python=python, threads=1L, batch_size=3L)
reference <- quantreg::boot.rq(X, y, tau=.5, R=9, cluster=g, U=V[g,,drop=FALSE])$B
delta <- max(abs(out$bootstrap_coefficients - reference))
stopifnot(delta < 1e-7, all(is.finite(out$standard_errors)))
print(out$coefficients)
print(out$standard_errors)
cat("Fixed quantreg draw comparison passed; maximum difference:", delta, "\n")
cat("Finite radii:", sum(is.finite(out$coefficient_radii)), "/", length(out$coefficient_radii), "\n")
