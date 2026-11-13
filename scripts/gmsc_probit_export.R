# GMSC probit GLM: R reference export
# Original R code: 'CRM :Credit Risk Modelling and Analysis' by Sunnyboy Ngobeni, v4 (Kaggle, Apache 2.0), copied unchanged except
# for data/output paths. The final block is added to export parity references.
# Run from the repository root:  Rscript scripts/<this file>
R.version.string
# Rscript cannot create a personal library on its own; use it so installs need no admin rights.
dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE, showWarnings = FALSE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
pkgs <- c("readxl", "readr", "ggplot2", "dplyr", "tidyr", "corrplot", "knitr", "caret", "pROC", "rmarkdown")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
pdf(NULL)  # discard plots when run with Rscript
data_dir <- Sys.getenv("GMSC_DIR", "data/GiveMeSomeCredit")
out <- Sys.getenv("OUT_DIR", "data/reference/gmsc_probit")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

# ---- chunk 1 ----
options(width = 700)
knitr::opts_chunk$set(warning = FALSE, message = FALSE, out.width = "100%", retina = 1)

# ---- chunk 2 ----
#header-includes:
  #- \usepackage{fvextra}
  #- \usepackage{pdflandscape}
  #- \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
  #the above is for changing the orientation of the pdf output to landscape for printing long df's or outputs
library(readxl) #4 importing .xls files
library(readr) #4 importing .csv files
library(ggplot2) #4 2d plots
library(dplyr) #4 data manipulation
library(tidyr) #4 data cleaning??
library(corrplot) #4 correlation test and plotting
library(knitr) #4 visualising tables and knitting documents
library(caret) #4 
library(pROC) #4 performance analytics i.e roc curve

# ---- chunk 3 ----
#Risk <- read_delim("Risk.csv", delim = ";", escape_double = FALSE, trim_ws = TRUE)
dictionary<-read_excel(file.path(data_dir, "Data Dictionary.xls"))
df_test<-read_csv(file.path(data_dir, "cs-test.csv"))
df_train<-read_csv(file.path(data_dir, "cs-training.csv"))
sample<-read_csv(file.path(data_dir, "sampleEntry.csv"))

# ---- chunk 4 ----
df_train<-rename(df_train,c(Applicant="...1",defaulted="SeriousDlqin2yrs",'unsecured.credit'=RevolvingUtilizationOfUnsecuredLines,age=age,'<2months Deliquency'=`NumberOfTime30-59DaysPastDueNotWorse`,DebtRatio=DebtRatio,MonthlyIncome=MonthlyIncome,openCredit=NumberOfOpenCreditLinesAndLoans,'3MonthsDefault'=NumberOfTimes90DaysLate,RealEstateLoans=NumberRealEstateLoansOrLines,'<3months Deliquency'=`NumberOfTime60-89DaysPastDueNotWorse`,Dependents=NumberOfDependents)) #from dplyr package, use to rename columns

colnames(df_test)<-colnames(df_train) #rename the given test dataset using the same name as the given train dataset.
str(df_train) #no. of columns and rows.
nrow(unique(df_train)) #checks for duplicates
#'[df_train<-unique(df_train)  #removes duplicates
colSums(is.na(df_train)) #checks for NAs per column
nrow(unique(df_test)) #checks for duplicates
colSums(is.na(df_test)) #checks for NAs per column

# ---- chunk 5 ----
rmarkdown::paged_table(head(df_train,5))
  # Kable_styling(latex_options="scale_down") OR
  # kable_styling(full_width=FALSE)%>%
  # column_spec(1,width="3cm")%>%
  # column_spec(2,width="5cm")
#The above commented out code scales down the kable output so it doesn't extend outside the page or makes column names overlap, can't seem to make it work yet.

# ---- chunk 6 ----
options(scipen = 999)
kable(summary(df_train))
rmarkdown::paged_table(df_train) #a pageable df in html, doesnt work 4 pdf.
df_train<-df_train|>filter(age>0)

# ---- chunk 7 ----
kable(head(df_train[,c(1,5,9,11)]%>%filter(`<2months Deliquency`==96),5))
kable(head(df_train[,c(1,5,9,11)]%>%filter(`<2months Deliquency`==98),5))

# ---- chunk 8 ----
df_train<-df_train%>%filter(`3MonthsDefault`<'96',DebtRatio<=1,unsecured.credit<=1)
df_test<-df_test%>%filter(`3MonthsDefault`<'96',DebtRatio<=1,unsecured.credit<=1)

kable(summary(df_train))

# ---- chunk 9 ----
par(mfrow=c(1,2))  #allows me to plot 2 plots side by side. 1 row, 2 columns.

pal<-c("coral2","mediumpurple","goldenrod","magenta","cyan","yellow","orange","green",rep("gray85",length(df_train$age)-8))

H1<-hist(df_train$age,main="train dataset: Distribution of age",xlab="Age",freq = FALSE,plot=FALSE)

H2<-hist(df_train$unsecured.credit,xlab="Utilization Of Unsecured credit",main="Distribution of unsecured credit utilization",ylab="No. of borrowers",plot=FALSE)

pal2<-rainbow(length(H1$counts))
pal3<-heat.colors(length(H2$counts))
hist(df_train$age,main="Distribution of age",xlab="Age",freq = FALSE,col=pal2)
lines(density(df_train$age),col="red")
hist(df_train$unsecured.credit,xlab="Utilization Of Unsecured credit",main="Unsecured credit utilization",ylab="No. of borrowers",col=pal3)

# ---- chunk 10 ----
kable(table(count=df_train$`<2months Deliquency`))
hist(df_train$DebtRatio,xlab="Debt to income ratio",main="Distribution of debt to income ratio",col=pal3)

# ---- chunk 11 ----
nrow(df_train%>%filter(MonthlyIncome>50000))

# ---- chunk 12 ----
s.income<-df_train%>%filter(MonthlyIncome<50000)
hist(s.income$MonthlyIncome,breaks=50,col=pal2,main="Distribution of Monthly income <50K",xlab="Monthly Income")

# ---- chunk 13 ----
par(mfrow=c(1,2))
m.income<-df_train%>%filter(MonthlyIncome>=50000&MonthlyIncome<=200000)
hist(m.income$MonthlyIncome,breaks=50,col=pal2,main="Monthly income >=50K and <=200K",xlab="Monthly Income")
l.income<-df_train%>%filter(MonthlyIncome>200000)
hist(l.income$MonthlyIncome,breaks=50,col=pal3,main="Monthly income >200K",xlab="Monthly Income")

# ---- chunk 14 ----

par(mfrow=c(1,2))

df_train%>%filter(defaulted==1)%>%group_by(age)%>%summarise(Total.defaults=n())%>%
  ggplot(aes(x=age,y=Total.defaults,color=age))+geom_point()+
  labs(title="Total defaults on loans by age")+
  scale_color_viridis_c()

df_train%>%filter(defaulted==0)%>%group_by(age)%>%summarise(Total.defaults=n())%>%
  ggplot(aes(x=age,y=Total.defaults,color=age))+geom_point()+
  labs(title="Total non-defaults on loans by age")

# ---- chunk 15 ----
boxplot(df_train$Dependents,ylab="number of dependents",col=pal3)

# ---- chunk 16 ----
kable(summary(df_train[,2:12]%>%filter(defaulted==1)))

# ---- chunk 17 ----
mydata<-df_train[,2:12] #only using colums 2 to 12
correlation<-select_if(mydata,is.numeric)%>%cor()
corrplot(correlation,type='upper',method="pie")

# ---- chunk 18 ----
logist<-glm(defaulted~.,data=mydata, family=binomial (link=probit))
varimp<-varImp(logist)

varimp["variable"]<-rownames(varimp)
varimp$variable<-factor(varimp$variable,levels = varimp$variable[order(varimp$Overall)])
varimp<-varimp%>%mutate(prop=Overall/sum(Overall)*100)%>%arrange(desc(prop))

pal<-c("coral2","mediumpurple","goldenrod",rep("blue1",length(varimp$prop)-3))
ggplot(varimp,aes(x=variable,y=Overall,))+
  labs(title="variable importance (overall and proportions)")+
  geom_bar(stat="identity",fill=pal)+
  geom_text(aes(label=round(prop,2)))+ #adds the proportions label to plot
  coord_flip()+ #flips the coordinates and makes the bar horizontal rather than vertical
  theme_minimal()

# ---- chunk 19 ----
df_train<-df_train%>%
  mutate(Dependents=replace_na(Dependents,0))%>%
  mutate(MonthlyIncome=replace_na(MonthlyIncome,median(MonthlyIncome,na.rm=TRUE)))
df_test<-df_test%>%
  mutate(Dependents=replace_na(Dependents,0))%>%
  mutate(MonthlyIncome=replace_na(MonthlyIncome,median(MonthlyIncome,na.rm=TRUE)))

# ---- chunk 20 ----
set.seed(123) #for reproducibility
df_train[,2]<-lapply(df_train[,2],factor) #making the default column factors/categorical variables.
#2,5,8,9,10,11,12
#df_train[,12]<-factor(df_train$Dependents,exclude=c(9,10,13,20))
split<-sample(1:nrow(df_train),0.7*nrow(df_train)) #random sampling
TrainData<-df_train[split,]
TestData<-df_train[-split,]

# ---- chunk 21 ----
logmod<-glm(defaulted~unsecured.credit+`<2months Deliquency`+`<3months Deliquency`+`3MonthsDefault`+age ,data=TrainData, family=binomial (link=probit)) #takes a lot of space/memory to rum, maybe making dependents into factors wasnt such a good idea. The final model didnt make use of dependents so it didn't matter.
summary(logmod)
pred<-predict(logmod,newdata=TestData,type="response")
#range(pred)
TestData$ProbOfDefault<-round(pred,5)
#PD<-data.frame(applicant=df_test$...1,round(pred,5))
#different cut-off probabilities: 15%, 18.5% and 20%

# ---- chunk 22 ----
pred_cut_0.04<-ifelse(TestData$ProbOfDefault>0.04,1,0)
pred_cut_0.1025<-ifelse(TestData$ProbOfDefault>0.1025,1,0)
pred_cut_0.2<-ifelse(TestData$ProbOfDefault>0.2,1,0)
TestData$pred_cut_0.04<-factor(pred_cut_0.04)
TestData$pred_cut_0.1025<-factor(pred_cut_0.1025)
TestData$pred_cut_0.2<-factor(pred_cut_0.2)

# ---- chunk 23 ----
par(mfrow=c(1,3))
kable(table(TestData$pred_cut_0.04))
kable(table(TestData$pred_cut_0.1025))
kable(table(TestData$pred_cut_0.2))

# ---- chunk 24 ----
ROC0.04<-roc(TestData$defaulted,pred_cut_0.04)
ROC0.1025<-roc(TestData$defaulted,pred_cut_0.1025)
ROC0.2<-roc(TestData$defaulted,pred_cut_0.2)
plot(ROC0.04)
lines(ROC0.1025,col="blue")
lines(ROC0.2,col="red")
legend.text1<-paste0("4% (AUC=", round(auc(ROC0.04),2),")")
legend.text2<-paste0("10.25% (AUC=", round(auc(ROC0.1025),2),")")
legend.text3<-paste0("20% (AUC=", round(auc(ROC0.2),2),")")
legend("bottomright", #position of legend
       legend=c(legend.text1,legend.text2,legend.text3),
       col=c("black","blue","red"), #curve colours
       lty=1, #line type (1=solid)
       lwd=2, #line width
       cex=0.75) #shrinks legend to 80% of its original size

# ---- chunk 25 ----
confusionMatrix(data=TestData$pred_cut_0.1025,reference = TestData$defaulted)
# positive=0, and negative=1

# ---- chunk 26 ----
confusionMatrix(data=TestData$pred_cut_0.04,reference = TestData$defaulted)
confusionMatrix(data=TestData$pred_cut_0.2,reference = TestData$defaulted)

# ---- chunk 27 ----
options(scipen = 999)
kable(summary(df_test))

# ---- chunk 28 ----
par(mfrow=c(1,2))
hist(df_test$age,main="Distribution of age",xlab="Age",freq = FALSE,col=pal2)
lines(density(df_test$age),col="red")
hist(df_test$unsecured.credit,xlab="Utilization Of Unsecured credit",main="Unsecured credit utilization",ylab="No. of borrowers",col=pal3)

# ---- chunk 29 ----
s.income<-df_test%>%filter(MonthlyIncome<50000)
hist(s.income$MonthlyIncome,breaks=50,col=pal2,main="Distribution of monthly income <50K",xlab="Monthly Income")

# ---- chunk 30 ----
par(mfrow=c(1,2))
m.income<-df_test%>%filter(MonthlyIncome>=50000&MonthlyIncome<=200000)
hist(m.income$MonthlyIncome,breaks=50,col=pal2,main="Monthly income >=50K and <=200K",xlab="Monthly Income")
l.income<-df_test%>%filter(MonthlyIncome>200000)
hist(l.income$MonthlyIncome,breaks=50,col=pal3,main="Monthly income >200K",xlab="Monthly Income")

# ---- chunk 31 ----
par(mfrow=c(1,2))
kable(table(days=df_test$`<2months Deliquency`))
plot(df_test$`3MonthsDefault`,df_test$`<3months Deliquency`,ylab="NumberOfTimes90DaysLate",xlab="NumberOfTime60-89DaysPastDueNotWorse")
boxplot(df_test$Dependents)

# ---- chunk 32 ----
prediction<-predict(logmod,newdata=df_test,type="response")
df_test$ProbOfDefault<-round(prediction,5)
df_test$defaulted<-factor(ifelse(df_test$ProbOfDefault>0.20,1,0))
#df_test$pred_cut_0.185<-factor(ifelse(df_test$ProbOfDefault>0.185,1,0))
#df_test$pred_cut_0.2<-factor(ifelse(df_test$ProbOfDefault>0.2,1,0))
kable(df_test%>%dplyr::select(Applicant,ProbOfDefault,defaulted)%>%tail(10))

# ---- chunk 33 ----

par(mfrow=c(1,2))

df_test%>%filter(defaulted==1)%>%group_by(age)%>%summarise(Total.defaults=n())%>%
  ggplot(aes(x=age,y=Total.defaults,color=age))+geom_point()+
  labs(title="Total defaults on loans by age")+
  scale_color_viridis_c()

df_test%>%filter(defaulted==0)%>%group_by(age)%>%summarise(Total.defaults=n())%>%
  ggplot(aes(x=age,y=Total.defaults,color=age))+geom_point()+
  labs(title="Total non-defaults on loans by age")

# ---- added: reference export ----
write.csv(data.frame(
  row_id      = TestData$Applicant,
  probability = as.numeric(predict(logmod, newdata = TestData, type = "response")),  # unrounded
  label       = as.integer(as.character(TestData$defaulted))),
  file.path(out, "holdout_predictions.csv"), row.names = FALSE)
write.csv(data.frame(row_id = TrainData$Applicant), file.path(out, "train_ids.csv"), row.names = FALSE)
co <- summary(logmod)$coefficients
write.csv(data.frame(term = rownames(co), co, check.names = FALSE),
          file.path(out, "coefficients.csv"), row.names = FALSE)
writeLines(c(paste("train_after_filter", nrow(df_train)), paste("train_rows", nrow(TrainData)),
             paste("holdout_rows", nrow(TestData))), file.path(out, "row_counts.txt"))
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
print(list.files(out))
