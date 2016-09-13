# Implement your own version of logistic regression solvers using gradient
# descent and Newton-Rhapson updates

logit <- function(z) {
  return(exp(z) / (1 + exp(z)))
}

cost <- function(X, y, beta) { ##return negative LL
  # a function that gives the value of the negative log-likelihood for specified
  # data and parameters
  # Args:
  #  X: a numeric matrix with rows representing observations and columns
  #  predictors
  #  y: a binary response vector with length equal to nrow(X)
  #  beta: a numeric vector of length ncol(X), giving the value of beta at which
  #  to evaluate the negative log-likelihood
  z <- X %*% beta
  LL <- t(y) %*% z-sum(log(1 + exp(z)))
  return (-LL)
}

gradient <- function(X, y, beta) {
  # a function that returns the gradient of the log-likelihood
  # Args:
  #  X: a numeric matrix with rows representing observations and columns
  #  predictors
  #  y: a binary response vector with length equal to nrow(X)
  #  beta: a numeric vector of length ncol(X), giving the value of beta at which
  #  to evaluate the gradient
  p <- exp(X %*% beta)/(1+exp(X %*% beta))
  score <- t(X) %*% (y - p)
  return (score)

}

hessian <- function(X, beta) {
  # a function that returns the Hessian of the log-likelihood
  # Args:
  #  X: a numeric matrix with rows representing observations and columns
  #  predictors
  #  y: a binary response vector with length equal to nrow(X)
  #  beta: a numeric vector of length ncol(X), giving the value of beta at which
  #  to evaluate the Hessian
  p <- exp(X %*% beta)/(1+exp(X %*% beta))
  w <- diag(c(p * (1-p)))
  J <- t(X) %*% w %*% X
  return (J)
}

fitModel_NR <- function(y, X, max.iter = 100, eps = 1e-10) {
  # a logistic regression solver using Newton-Rhapson updates to find beta
  # Args:
  #  y: a binary response vector with length equal to nrow(X)
  #  X: a numeric matrix with rows representing observations and columns
  #  predictors
  #  max.iter: the maximum number of updates to perform, if beta does not
  #  converge before this iteration, indicate non-convergence
  #  eps: convergence threshold. If average absolute distance between beta[t]
  #  and beta[t+1] is below eps, stop updating
  # Return:
  #  cost: a numeric vector giving the value of the negative log-likelihood at 
  #  each iteration
  #  beta: a numeric vector giving the estimate of beta at time t (either
  #  max.iter or after convergence critereon is reached)
  #  betas: a numeric matrix indicating the estimates of beta at iterations
  #  1:(t-1)   
  n.beta <- ncol(X)
  B <-  matrix(NA,ncol = max.iter,nrow = n.beta)
  NLL <- rep(0,max.iter) ###negative LL
  B[,1] <- rep(0,ncol(X)) ##starting value
  NLL[1] <- cost(X,y,B[,1])
  for (i in 2:max.iter) {
    B[,i] <- B[,i-1]+solve(hessian(X,B[,i-1])) %*% gradient(X,y,B[,i-1])
    NLL[i] <- cost(X,y,B[,i-1])
    if (all(abs(B[,i]-B[,i-1]) < eps)) break;
  }
  B <- B[ , !apply(is.na(B), 2, all)] ##remove NA columns if there is any 
  NLL <- NLL[NLL>0] 
  return (list(B,NLL))
}


fitModel_GD <- function(y, X, max.iter = 10000, eps = 1e-10) {
  # a logistic regression solver using gradient descent updates to find beta
  # Args:
  #  y: a binary response vector with length equal to nrow(X)
  #  X: a numeric matrix with rows representing observations and columns
  #  predictors
  #  max.iter: the maximum number of updates to perform, if beta does not
  #  converge before this iteration, indicate non-convergence
  #  eps: convergence threshold. If average absolute distance between beta[t]
  #  and beta[t+1] is below eps, stop updating
  # Return:
  #  cost: a numeric vector giving the value of the negative log-likelihood at 
  #  each iteration
  #  beta: a numeric vector giving the estimate of beta at time t (either
  #  max.iter or after convergence critereon is reached)
  #  betas: a numeric matrix indicating the estimates of beta at iterations
  #  1:(t-1)
  n.beta <- ncol(X)
  B <-  matrix(NA,ncol = max.iter,nrow = n.beta)
  NLL <- rep(0,max.iter) ###negative LL
  B[,1] <- rep(0,ncol(X)) ##starting value
  NLL[1] <- cost(X,y,B[,1])
  alpha <- 4/(svd(cbind(1, X))$d[1]^2)
  
  for (i in 2:max.iter) {
    B[,i] <- B[,i-1]+alpha*gradient(X,y,B[,i-1])
        NLL[i] <- cost(X,y,B[,i-1])
        if (all(abs(B[,i]-B[,i-1]) < eps)) break;
  }
  B <- B[ , !apply(is.na(B), 2, all)] ##remove NA columns if there is any 
  NLL <- NLL[NLL>0] 
  return (list(B,NLL))

}

