# Test Survival Models in R
# eg. http://www.uni-kiel.de/psychologie/rexrepos/posts/survivalCoxPH.html
# http://www.ms.uky.edu/~mai/Rsurv.pdf

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Simulate Data # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
set.seed(1)
N = 1000	# number of samples
p = 2		# dimension baseline covariates
x = matrix(runif(n=N*p, min=-1, max=1), ncol=p)	# baseline covariates
tf = 15 + 5*runif(n=N) 	# max follow up time
y0 = rexp(n=N,rate=0.5)
beta = matrix(c(1,0))
y = y0 * exp(-x %*% beta)    # death time, according to cox exponential model
d = y < tf		# event indicator
t = pmin(y,tf)	# observation time
dfSurv <- data.frame(t, d, x)
print(dfSurv)
z <- x %*% beta

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# library(survival)
library(rms)

rightside <- paste(names(dfSurv[,3:(p+2)]), collapse="+")   # up to first order interactions
fmla <- as.formula(paste("Surv(t, d) ~ ", rightside))
print(fmla)
fit <- cph(formula = fmla, data=dfSurv) 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

predict(coxfit)


# pred_test = rfSRC.predict_rfsrc(rfsc,test)
# print 'Test C-Index:', 1 - pred_test.rx('err.rate')[0][-1]
# predict(treefit, x.test)

