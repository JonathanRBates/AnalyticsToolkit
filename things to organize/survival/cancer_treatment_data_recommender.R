# Keep variables/mappings similar to example in 
#   Royston, Altman. "External validation of a cox prognostic model: principles and methods"
# Prognostic Factors: 
#   tumor size, 
#   menopausal status, 
#   hormone treatment, 
#   age at surgery (years), 
#   number of positive lymph nodes, 
#   progesterone receptors, fmol/l, 
#   oestrogen receptors, fmol/l
# Outcome: Recurrence free survival time
# 

# install.packages("plyr")
# library("plyr")

x.train <- read.table(file = "cancer_treatment_train.csv", sep = ",", header=TRUE)
x.test <- read.table(file = "cancer_treatment_test.csv", sep = ",", header=TRUE)

# normalizations made by Royston-Altman
x.train$age <- x.train$age/100
x.test$age <- x.test$age/100
x.train$er <- x.train$er/1000
x.test$er <- x.test$er/1000

# Check
print(summary(x.train))
print(summary(x.test))

library(survival)

fit <- survfit(Surv(time, event) ~ 1, data=x.test, conf.type='log', conf.int=0.95)
plot(fit, xlab="Time", ylab="Survival Probability")

fit.treat <- survfit(Surv(time, event) ~ treat, data=x.test, conf.type='log', conf.int=0.95)
# plot(fit.treat, xlab="Time", ylab="Survival Probability", mark.time=F, col=1:2, lwd=2, conf.int=T)
# legend("topright",legend=c('untreated','treated'),col=1:2,horiz=FALSE,bty='n')

# install.packages("survminer")
library("survminer")
ggsurvplot(fit.treat,  size = 1,  # change line size
           linetype = "strata", # change line type by groups
           break.time.by = 250, # break time axis by 250
           palette = c("#E7B800", "#2E9FDF"), # custom color palette
           conf.int = TRUE, # Add confidence interval
           pval = TRUE # Add p-value
)

# two-sample log-rank
survdiff(Surv(time, event)~treat, data=x.test)  



# DEFINE FORMULAS FOR VARIOUS MODEL SPECIFICATIONS 
rightside <- rep("",4)
rightside[1] <- paste(names(x.train[,4:ncol(x.train)]), collapse="+")   # up to first order interactions
rightside[2] <- paste("(",rightside[1],")^2")   # up to second order interactions
rightside[3] <- paste("treat*(",rightside[1],")")   # first order, and treatment interactions
rightside[4] <- "treat+as.factor(size)+meno+I(age^3):log(age)+I(age^3)+I(nodes^(-0.5))+er"

regfit <- vector(mode="list", length=length(rightside))
coxfit <- vector(mode="list", length=length(rightside))
for (ii in 1:length(rightside)){
  fmla <- as.formula(paste("Surv(time, event) ~ ", rightside[ii]))
  print(fmla)
  # regfit[[ii]] <- survreg(fmla, data=x.train, dist="exponential")
  coxfit[[ii]] <- coxph(fmla, data=x.train)  
}

for (ii in 1:length(rightside)){
  print(coxfit[[ii]]$formula)
  print(coxfit[[ii]]$loglik)
  print(summary(coxfit[[ii]])$concordance)
}

print('Printing Royston-Altman model...')
print(summary(coxfit[[4]]))


coxph.recommend <- function (coxfit, x){
  # coxfit is a fit object returned from coxph()
  # x is a data frame with a column labeled 'treat';
  #  the treat column is a binary "treatment" with values 0/1
  x.1 <- x
  x.0 <- x
  x.1$treat <- 1
  x.0$treat <- 0
  out.1 <- predict(coxfit, x.1, type="lp")
  out.0 <- predict(coxfit, x.0, type="lp")
  recommended <- max.col(cbind(out.1,out.0))-1
  return(as.factor(recommended))
}

for (ii in 1:length(rightside)){
  print(coxfit[[ii]]$formula)
  recommended <- coxph.recommend(coxfit[[ii]], x.test)
  print(summary(recommended))
}




# RANDOM SURVIVAL FORESTS

install.packages('randomForestSRC')
library('randomForestSRC')
treefit <- rfsrc(Surv(time, event)~., data=x.train, ntree=100)
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