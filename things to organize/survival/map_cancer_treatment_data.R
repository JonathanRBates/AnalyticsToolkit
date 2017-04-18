#
# Code used to extract and preprocess datasets for validating a Treatment Recommender System in
# 
#   Deep Survival: A Deep Cox Proportional Hazards Network (2016)
#
#   Jared Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, Yuval Kluger
# 
#   Yale University
#
#   https://arxiv.org/abs/1606.00931
#
#
# Inspired by Royston, Altman. "External validation of a cox prognostic model: principles and methods"
#

# GET PACKAGE FOR ROTT2 DATA
install.packages('AF')
library("AF")

# GET PACKAGE FOR GBSG2 DATA
install.packages("TH.data")
library("TH.data")

data("rott2") # (Training/Derivation) Rotterdam tumor data
data("GBSG2") # (Testing/Validation) German Breast Cancer Study Group
x.train <- rott2
x.test <- GBSG2
names(x.train)
print(head(x.train, n=5))
print(head(x.test, n=5))

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

install.packages("plyr")
library("plyr")

# Remap training set
help(rott2)
print(unique(x.train$size))
x.train$size <- revalue(x.train$size, c("<=20mm"=0.,">20-50mmm"=1.,">50mm"=2.))
print(unique(x.train$meno))
x.train$meno <- revalue(x.train$meno, c("pre"=0.,"post"=1.))
print(unique(x.train$hormon))
x.train$treat <- revalue(x.train$hormon, c("no"=0.,"yes"=1.))
x.train <- x.train[x.train$nodes > 0,]    # remove node-negative
x.train$event <- apply(x.train, 1, function(u) {u['rfi']==1 | u['osi']=='deceased'} )
print(sum(x.train$event)==1080)
x.train$time <- apply(x.train[,c('rf','os')], 1, min)   # recurrence free survival time
x.train$event[x.train$time>=84] = FALSE
x.train$time[x.train$time>84] = 84
print(sum(x.train$event))
head(x.train[,c('rf','rfi','mf','mfi','os','osi','time','event')],n=100)
x.train <- x.train[,c('event','time','treat','size','meno','age','nodes','pr','er')]
head(x.train)

# Remap test set
help(GBSG2)
# print(unique(x.test$tsize))
x.test$size <- cut(x.test$tsize, breaks=c(-Inf,20,50,Inf), labels=c(0.,1.,2.))
print(unique(x.test$menostat))
x.test$meno <- revalue(x.test$menostat, c("Pre"=0.,"Post"=1.))
print(unique(x.test$horTh))
x.test$treat <- revalue(x.test$horTh, c("no"=0.,"yes"=1.))
# x.test$age
x.test$nodes <- x.test$pnodes
x.test$pr <- x.test$progrec
x.test$er <- x.test$estrec
head(x.test)
x.test$event <- x.test$cens==1
x.test$time <- x.test$time/30.4375
x.test <- x.test[,c('event','time','treat','size','meno','age','nodes','pr','er')]
head(x.test)

# Check
print(summary(x.train))
print(summary(x.test))

write.table(x.train, file = "cancer_treatment_train.csv", sep = ",", col.names = NA, qmethod = "double")
write.table(x.test, file = "cancer_treatment_test.csv", sep = ",", col.names = NA, qmethod = "double")
