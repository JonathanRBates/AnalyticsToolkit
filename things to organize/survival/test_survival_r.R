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
library(survival)
library(rms)

# KM curves
fit1 <- survfit(Surv(t,d) ~ 1, data=dfSurv)
# fit1 <- survfit(Surv(t,d) ~ 1)
plot(fit1, xlab="Time", ylab="Survival Probability")   # FOR EXPONENTIAL COX MODEL, SHOULD BE exp(-rate*t)

# fmla1 <- as.formula(paste("Surv(t, d) ~ ", paste(names(dfSurv[,3:(p+2)]), collapse="+")))
# regfit1 <- survreg(fmla1, data=dfSurv, dist="exponential")
# print(regfit1)

rightside <- rep("",2)
rightside[1] <- paste(names(dfSurv[,3:(p+2)]), collapse="+")   # up to first order interactions
rightside[2] <- paste("(",rightside[1],")^2")   # up to second order interactions

regfit <- vector(mode="list", length=length(rightside))
coxfit <- vector(mode="list", length=length(rightside))
coxfit2 <- vector(mode="list", length=length(rightside))
for (ii in 1:length(rightside)){
  fmla <- as.formula(paste("Surv(t, d) ~ ", rightside[ii]))
  print(fmla)
  regfit[[ii]] <- survreg(fmla, data=dfSurv, dist="exponential")
  coxfit[[ii]] <- coxph(fmla, data=dfSurv)  
  coxfit2[[ii]] <- cph(fmla, data=dfSurv, surv=TRUE)
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# PLOT
# 
# library(ggplot2)

# plot(dfSurv[,c('X1','X2')], xlab="X label", ylab="Y label", pch=19, cex=.4)
# contour(z=y, drawlabels=FALSE, nlevels=5, col=my.cols, add=TRUE)

# install.packages('akima')
library('akima')

df <- data.frame(x,z)
fld <- with(df, interp(x = X1, y = X2, z = z))

filled.contour(x = fld$x,
               y = fld$y,
               z = fld$z,
               color.palette = colorRampPalette(c("white", "blue")),
               xlab = "X1",
               ylab = "X2",
               main = "True Risk", 
               cex.main = 1)

df <- data.frame(x,Z=predict(coxfit[[1]], dfSurv, type='lp'))
fld <- with(df, interp(x = X1, y = X2, z = Z))

filled.contour(x = fld$x,
               y = fld$y,
               z = fld$z,
               color.palette = colorRampPalette(c("white", "blue")),
               xlab = "X1",
               ylab = "X2",
               main = "Predicted Risk (Survival, linear)", 
               cex.main = 1)


library(pec)
z.fromSurvival <- predictSurvProb(coxfit2[[1]], newdata=dfSurv, times = 1)
df <- data.frame(x,Z=-log(z.fromSurvival))
fld <- with(df, interp(x = X1, y = X2, z = Z))

filled.contour(x = fld$x,
               y = fld$y,
               z = fld$z,
               color.palette = colorRampPalette(c("white", "blue")),
               xlab = "X1",
               ylab = "X2",
               main = "Predicted Risk (RMS/PEC, linear)", 
               cex.main = 1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# RANDOM SURVIVAL FORESTS

# install.packages('randomForestSRC')
library('randomForestSRC')
treefit <- rfsrc(Surv(t, d)~., data=dfSurv, ntree=100)

df <- data.frame(x,Z=predict(treefit, dfSurv)$predicted)
fld <- with(df, interp(x = X1, y = X2, z = Z))
# fld <- with(df, interp(x = X1, y = X2, z = predict(treefit, dfSurv)$predicted))


filled.contour(x = fld$x,
               y = fld$y,
               z = fld$z,
               color.palette = colorRampPalette(c("white", "blue")),
               xlab = "X1",
               ylab = "X2",
               main = "Predicted Risk (random forest)", 
               cex.main = 1)












predict(treefit, x.test)


tree.recommend <- function (treefit, x){
  # coxfit is a fit object returned from coxph()
  # x is a data frame with a column labeled 'treat';
  #  the treat column is a binary "treatment" with values 0/1
  x.1 <- x
  x.0 <- x
  x.1$treat <- 1
  x.0$treat <- 0
  out.1 <- predict(treefit, x.1)
  out.0 <- predict(treefit, x.0)
  recommended <- max.col(cbind(out.1,out.0))-1
  return(as.factor(recommended))
}

recommended <- tree.recommend(treefit, x.test)
summary(recommended)


