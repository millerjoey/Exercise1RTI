# Start
library("dplyr")
library("ggplot2")
library("bayesglm")

setwd("Desktop/RTI")
records <- read.table("records.csv", header = F, sep=",")
# Assumes the table created in the .SQL file.

# Renaming columns. Did not do that when I exported from sqlite:
names(records) <- c("id", "age", "workclass", "education_level", "education_num", 
                    "marital_status", "occupation", "TBD", "race", "sex", 
                    "capital_gain", "capital_loss", "hours_week", "country", 
                    "over_50k")

#==================Training/Test==================
# Split 80/20 training/test:
set.seed(12345)
sampRows <- sample(x = 1:48842, size=round(0.8*48842), replace=F)
train <- records[sampRows,]
test <- records[-sampRows,]
rm(records)

#==============================================
#=======Exploring and Feature Selection========
#==============================================
apply(train, 2, function(x) sum(is.na(x)))
# No NA-type missing data.

apply(train, 2, function(x) sum(x=="?"))
# Occupation, workclass, country, have missing values. Will try calling "?" a legitimate class.

#===============First-Order Effects============
# Only 14 covariates; two of which were collinear.
# The section below generates plots, considers and
# justifies transformations.

#-----------------||AGE||----------------------
# Age, as histogram, colored by mean(over_50k):
grp <- group_by(train, age)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=age, y=count, fill=Y,  color=Y)) +
  geom_bar(stat="identity") +
  scale_fill_continuous(name="Prob") +
  scale_color_continuous(name="Prob") +
  scale_y_log10(breaks=c(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)) + 
  scale_x_continuous(breaks=seq(min(train$age),max(train$age), by=5)) +
  labs(title="Histogram of Age shaded by Probability")

dev.print(pdf, "HistAgeProb.pdf")

# Clearly not increasing (in its marginal distribution). 
# Will convert to Factor for simplicity. Might want to
# see if probability is increasing in Age|X for some other
# covariates.

trainCut <- cbind.data.frame(train, "ageCut"=cut(train$age, breaks = c(16, 23, 27, 35, 55, 62, 67, 95)))
View(trainCut)
grp <- group_by(trainCut, ageCut)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=ageCut, y=Y)) +
  geom_bar(stat="identity") +
  labs(title="P(Y=1) by Age")
# These are relatively good ways to cut the ages.

#-------------||OCCUPATION||--------------
# mean(over_50k) averaged by occupation with sample sizes:
grp <- group_by(trainCut, occupation)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=occupation, y=Y)) +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_point() +
  geom_text(label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by occupation (with sample sizes)")
# Lots of variation, as expected, by job title. Can justify an "(if occupation=="Priv-house-serv") {Y=0} line":
filter(df, occupation=="Priv-house-serv") # 0.0124, (n=242)

#-------------||WORKCLASS||--------------
grp <- group_by(trainCut, workclass)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=workclass, y=Y)) +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_point() +
  geom_text(label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by Workclass (with sample sizes)")

#--------||EDUCATION LEVEL/NUM||--------------
grp <- group_by(trainCut, education_level, education_num)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=education_num, y=Y)) +
  geom_point() +
  geom_text(label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by Education Level (with sample sizes)", x="Education Num/Level")
# Call View(grp) to see that education_level and education_num are collinear.


# Not a large difference between 4-8. Will bin them in a new column and transform:
educationCut=cut(train$education_num, breaks = c(0.5,1.5,2.5,3.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5, 16.5))
train <- cbind.data.frame(trainCut, "educationCut"=as.integer(educationCut))

# Looking again, we have a monotonic relationship and more DOF leftover
# for interactions. It turns out, this also helped to make some trends
# more consistent for Age|Education level.
grp <- group_by(train, educationCut)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=educationCut, y=Y)) +
  geom_point() +
  geom_text(label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by Education Level (with sample sizes)", x="educationCut")

#-------------||MARITAL STATUS||--------------
grp <- group_by(train, marital_status)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=marital_status, y=Y)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_text(label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by Marital Status (with sample sizes)")
# Interactions by male/female turned out to be an important
# aspect of the model.

#-------------||RACE||--------------
grp <- group_by(train, race)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=race, y=Y)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_text(label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by Race (with sample sizes)")
# Expect that much of this variation is explained by country.

#-------------||SEX||--------------
grp <- group_by(train, sex)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=sex, y=Y)) +
  geom_bar(stat="identity", fill="grey", color="black") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_text(label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by sex (with sample sizes)")

#-------------||CAPITAL GAIN||--------------
sum(train$capital_gain==0)
# 44,000 zeros. Complicates things. If I have time, make two models with and without
# capital_gain and an if (capital_gain==0) clause.

ggplot(data=train, aes(x=log(1+capital_gain), y=over_50k)) +
  geom_point() +
  labs(title="log(1+capital_gain) and Y") +
  geom_jitter()
# Looks perfect for Logit if I ignore the zero cap gains.

dim(filter(train, capital_gain>0))

#-------------||CAPITAL LOSS||--------------
sum(train$capital_loss==0)
# 46,000 zeros. See capital gain.

ggplot(data=train, aes(x=log(1+capital_loss), y=over_50k)) +
  geom_point() +
  labs(title="log(1+capital_loss) and Y") +
  geom_jitter()
# Less useful than cap gain.

#-------------||NET CAPITAL GAIN||--------------
ggplot(data=train, aes(x=log(1+capital_gain-capital_loss), y=over_50k)) +
  geom_point() +
  labs(title="log(1+capital_loss) and Y") +
  geom_jitter()
# 2282 had more loss than gains. Don't bother.

#-------------||HOURS WEEK||--------------

grp <- group_by(train, "hours_week"=cut(hours_week, breaks=seq(0, 100, by=5)))
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=hours_week, y=count, fill=Y,  color=Y)) +
  geom_bar(stat = "identity") +
  theme(axis.text.x=element_text(angle=45, hjust=1)) +
  scale_fill_continuous(name="Prob") +
  scale_x_discrete(labels=c("0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80", "81-85", "86-90", "91-95", "96-100")) + 
  scale_y_log10(breaks=c(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4096*2, 4096*4)) +
  scale_color_continuous(name="Prob") +
  labs(title="Histogram of Hours per Week colored by Probability", x="hours per week")
# Isn't monotonic. Cut arbitrarily into 10 groups for more
# sample size, look at interactions.
dev.print(pdf, "HistHoursProb.pdf")

hoursCut=cut(train$hours_week, breaks = c(0, 10, 20, 30, 40,50,60,70,80,90,100))
train <- cbind.data.frame(train, hoursCut)

grp <- group_by(train, hoursCut)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
ggplot(data=df, aes(x=hoursCut, y=Y)) +
  geom_bar(stat="identity", fill="grey", color="black") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_text(size=3, label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by hours per week (with sample sizes)")
#-------------||COUNTRY||--------------
grp <- group_by(train, country)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
df$country <- reorder(df$country, df$Y)
ggplot(data=df, aes(x=country, y=Y)) +
  geom_bar(stat="identity", fill="grey", color="black") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_text(size=3, label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by Country (with sample sizes)")
# Want to cut into a few different groups. First, let's see
# if there's interaction with races:

grp <- group_by(train, country, race)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
df$country <- reorder(df$country, df$Y)
ggplot(data=df, aes(x=country, y=Y)) +
  geom_bar(stat="identity", fill="grey", color="black") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  # geom_text(size=3, label=df$count, nudge_y = .03) +
  facet_grid(race~.) +
  labs(title="P(Y=1) by Country")
# Ordered by race x country unhelpfully. Nevertheless, it's clear 
# that bars are not all ordered identically upon separating by 
# race. This suggests an interaction.
# Note: Geom_text is causing trouble with the sorting. Will do more
# detailed interaction plots later.

# I'll add a column for (roughly) continents:
continents <- function(x) {
  asia <- c("vietnam", "laos", "thailand", "hong", "philippines", "china", "cambodia", "japan", "taiwan", "india", "iran")
  europe <- c("holand-netherlands", "scotland", "poland", "portugal", "germany", "ireland", "greece", "england", "france", "italy", "yugoslavia")
  englishAmerica <- c("united-states", "canada")
  spanishAmerica <- c("mexico", "guatemala", "columbia", "nicaragua", "dominican-republic", "el-salvador", "peru", "puerto-rico", "ecuador", "cuba")
  unknown <- c("?")
  x <- tolower(x)
  if (x %in% asia) {return("asia")}
  if (x %in% europe) {return("europe")}
  if (x %in% englishAmerica) {return("english-america")}
  if (x %in% spanishAmerica) {return("spanish-america")}
  if (x %in% unknown) {return("?")}
  else {return("other")}
}
continent <- unlist(sapply(train$country, continents))
train <- cbind.data.frame(train, continent)

#-------------||CONTINENT||--------------
grp <- group_by(train, continent)
df <- summarize(grp, "Y"=mean(over_50k), "count"=n())
df$country <- reorder(df$continent, df$Y)
ggplot(data=df, aes(x=continent, y=Y)) +
  geom_bar(stat="identity", fill="grey", color="black") +
  theme(axis.text.x = element_text(angle = 45, hjust=1)) +
  geom_text(size=3, label=as.character(df$count), nudge_y = .03) +
  labs(title="P(Y=1) by Continent (with sample sizes)")

#==============Second-Order Effects=====================

# This will generate every combination of interaction plots.
# Two calls to ggplot are because I found the 36 combination
# via combn() but want to swap x-y axes and make them next to
# each other in the list.
trainF <- cbind(train, "educationFactor"= as.factor(train$educationCut))
list <- combn(names(trainF)[-c(1, 2, 4, 5, 8, 11, 12, 13, 14, 15, 17)], 2, simplify=F)
for (i in list) {
  grp <- group_by_(trainF, (i[1]), (i[2]))
  df <- summarize(grp, "Y"=mean(over_50k), "SEtop"=mean(over_50k)+1.96*mean(over_50k)*(1-mean(over_50k))/sqrt(n()), 
                  "SEbottom"=mean(over_50k)-1.96*mean(over_50k)*(1-mean(over_50k))/sqrt(n()), count=n())
  print(ggplot(data=df, aes_string(x=(i[1]), y="Y", color=i[2])) +
          theme(axis.text.x = element_text(angle = 45, hjust=1)) +
          geom_line(aes_string(group=i[2])) +
          geom_point(aes_string(group=i[2])) +
          #geom_errorbar(aes(ymax=SEtop, ymin=SEbottom)) + # Too messy.
          labs(title=paste("P(Y=1) in", i[1], "x", i[2])))
  i <- i[c(2,1)]
  grp <- group_by_(trainF, (i[1]), (i[2]))
  df <- summarize(grp, "Y"=mean(over_50k), "SEtop"=mean(over_50k)+1.96*mean(over_50k)*(1-mean(over_50k))/sqrt(n()), 
                  "SEbottom"=mean(over_50k)-1.96*mean(over_50k)*(1-mean(over_50k))/sqrt(n()), count=n())
  print(ggplot(data=df, aes_string(x=(i[1]), y="Y", color=i[2])) +
          theme(axis.text.x = element_text(angle = 45, hjust=1)) +
          geom_line(aes_string(group=i[2])) +
          geom_point(aes_string(group=i[2])) +
          #geom_errorbar(aes(ymax=SEtop, ymin=SEbottom)) + # Too messy.
          labs(title=paste("P(Y=1) in", i[1], "x", i[2])))
}


# Note: Used new data frame "trainF" to convert education to a factor. Was being
# evaluated with continuous colors; couldn't change ggplot's default easily.

# Optionally: Add ggplot(data=filter(df, count>30), aes(...)...) 
# to eliminate the noise or add SE bars. But beware: the lines 
# will just skip the points and connect, so it may
# look like more line crossings. SE bars were too messy.

# Discussion and TODO:
# I generated all pairwise plots to look for interactions. A more thorough
# analysis might use the SE bars (or correlation, etc) that I made to judge 
# more objectively which factors likely interact. However, it's not 
# so simple because the data will all be transformed and there's no easy 
# way to judge the relative impact of improbable interaction effects of
# different magnitudes on the P(Y=1) scale.
# Alternatively, would also be interesting to put priors on the interactions 
# and get the final predictions by sampling from the posterior distribution 
# of each Beta in the model.

# Important interactions can be recognized by intersections. On this
# scale, line segment slopes cannot necessarily indicate interactions, but
# are suggestive.


# Below, I looked through the charts generated, made some comments, and started
# haphazardly training the model to see which variables fit the training set.
# Much more work can be done.

#----------||AGE x EDUCATION||-----------
# Conclusion: Age seems to interact with only a few highest grade levels.
# not a priority to include for interactions.


#----------||OCCUPATION x WORKCLASS||-----------
# Looking further into the breakdown of each occupation by workclass. Generates a plot
# with standard errors (of the mean) once for each occupation.
for (i in levels(train$occupation)) {
  grp <- group_by(filter(train, occupation==i),  workclass)
  df <- summarize(grp, "Y"=mean(over_50k), "SEtop"=mean(over_50k)+1.96*mean(over_50k)*(1-mean(over_50k))/sqrt(n()), 
                  "SEbottom"=mean(over_50k)-1.96*mean(over_50k)*(1-mean(over_50k))/sqrt(n()), count=n())
  print(ggplot(data=df, aes(x=workclass, y=Y)) +
          theme(axis.text.x = element_text(angle = 45, hjust=1)) +
          geom_point() +
          geom_errorbar(aes(ymax=SEtop, ymin=SEbottom)) +
          labs(title=paste("P(Y=1) in", i, "\nby Workclass"), y="P(Y=1)"))
}
# Looks like the patterns are similar (when available). Interactions is not a priority.

#----------||MARITAL_STATUS x SEX||-----------
# Can justify interaction term.

#----------||CONTINENT x SEX||-----------
# Not a priority

#----------||CONTINENT x RACE||-----------
# Yes, use this.

#----------||MARITAL_STATUS x RACE||-----------
# No strong reason.

#----------||HOURSCUT x CONTINENT||---------------
# Nothing interesting.

#===================Training the Models=========================

fit <- glm(data=train, formula = 
             over_50k ~ 
             log(1+capital_gain) + log(1+capital_loss) +
             marital_status + occupation +
             educationCut*race + hoursCut + continent*race +
             sex*marital_status + ageCut*occupation
           ,
           family=binomial(link="logit"))

preds <- predict.glm(object = fit, type = "response", newdata = train)

sum(ifelse(preds>0.5, 1, 0))
sum(ifelse(preds>0.5, 1, 0)==train$over_50k)/39074
# Accuracy: 85.2% on training data.

# Problem: bayesglm (gets point estimates for the coefs, but allows for a prior)
# gets radically different, even with noninformative prior. With improper prior,
# fails to converge. TODO: the math to see that posterior is improper and find
# better priors, possibly a good idea to get full distributions with MCMC.

# fitb <- bayesglm(data=train, formula =
#                    over_50k ~
#                    log(1+capital_gain) + log(1+capital_loss) +
#                    marital_status + occupation +
#                    educationCut*race + hoursCut + continent*race +
#                    sex*marital_status + ageCut*occupation
#                  , family=binomial(link="logit"), prior.scale = Inf)
# predsb <- predict.glm(object = fitb, type = "response", newdata = train)
# sum(ifelse(predsb>0.5, 1, 0))
# sum(ifelse(predsb>0.5, 1, 0)==train$over_50k)/39074


sum(!train$over_50k)/39074 # 'Null"  model. Predicts zero for everyone.
                           # Accuracy: 76.2%

#==================Prediction Function==========================

prediction <- function(newdata, model) {
  # Accepts a data frame with names(newdata)="id", "age", "workclass", "education_level", "education_num", 
  #"marital_status", "occupation", "TBD", "race", "sex", 
  #"capital_gain", "capital_loss", "hours_week", "country"
  # Will generate and transform the appropriate columns.
  
  # Transformations:
  ageCut <- cut(newdata$age, breaks = c(16,23,27,35,55,62,67,95))
  educationCut <- as.integer(cut(newdata$education_num, breaks = c(0.5,1.5,2.5,3.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5, 16.5)))
  continent <- unlist(sapply(newdata$country, continents))
  hoursCut <- cut(newdata$hours_week, breaks = c(0,10,20,30,40,50,60,70,80,90,100))
  
  # Bind to data frame
  newdata <- cbind.data.frame(newdata, hoursCut, ageCut, educationCut, continent)
  
  # Use 0-1 loss. Returns vector of 1s and 0s rounded.
  preds <- predict.glm(object = model, type = "response", newdata)
  return(ifelse(preds>0.5, 1, 0))
}

sum(prediction(test, fit)==test$over_50k)/9768
                # Final Accuracy: 84.3%
