rm(list = ls())

# install the environment 
library(tidyverse)
library(caret)
library(car)

# read the data
installment2_id01 = read.csv("installment2_id01.csv")

# make NAICS, WomanOwned, FICO categorical variables

installment2_id01 = installment2_id01 %>%
  mutate(NAICS = as.factor(NAICS),
         WomanOwned = as.factor(WomanOwned),
         CorpStructure = as.factor(CorpStructure),
         FICO = case_when(
           (300 <= FICO)&(FICO <= 579) ~ "Poor",
           (580 <= FICO)&(FICO <= 669) ~ "Fair",
           (670 <= FICO)&(FICO <= 739) ~ "Good",
           (740 <= FICO)&(FICO <= 799) ~ "Very Good",
           (800 <= FICO)&(FICO <= 850) ~ "Excellent",
         ))

model_rm_outliers = lm(PRSM ~. , data = installment2_id01)
installment2_id01$residual=residuals(model_rm_outliers)
plot(model_rm_outliers,1)

# remove 11 outliers observed in the plot (abs(residual)>=1)
installment2_id01 = installment2_id01 %>% 
  filter(abs(residual)<=1) %>% 
  dplyr::select(-residual) 

# TODO: remove after certain months

installment2_id01 = installment2_id01 %>% 
  filter(abs(Months)<=60)

# numeric: TotalAmtOwed, Volume, Stress, Num_Delinquent, Num_CreditLines, Months

plot(installment2_id01$PRSM, installment2_id01$TotalAmtOwed); lines(lowess(installment2_id01$PRSM, installment2_id01$TotalAmtOwed))
plot(installment2_id01$PRSM, installment2_id01$Volume)
plot(installment2_id01$PRSM, installment2_id01$Stress)
plot(installment2_id01$PRSM, installment2_id01$Num_Delinquent)
plot(installment2_id01$PRSM, installment2_id01$Num_CreditLines)
plot(installment2_id01$PRSM, installment2_id01$Months)

model_total = lm(PRSM ~ TotalAmtOwed, data = installment2_id01)
plot(installment2_id01$TotalAmtOwed, residuals(model_total),
     xlab = "TotalAmtOwed", ylab = "Residuals") +
lines(lowess(installment2_id01$TotalAmtOwed, residuals(model_total))) +
abline(h = 0, lty = 2)

model_volume = lm(PRSM ~ Volume, data = installment2_id01)
plot(installment2_id01$Volume, residuals(model_volume),
     xlab = "Volume", ylab = "Residuals") + 
  lines(lowess(installment2_id01$Volume, residuals(model_volume))) +
  abline(h = 0, lty = 2)

model_Stress = lm(PRSM ~ Stress, data = installment2_id01)
plot(installment2_id01$Stress, residuals(model_Stress),
     xlab = "Stress", ylab = "Residuals") +
lines(lowess(installment2_id01$Stress, residuals(model_Stress))) +
abline(h = 0, lty = 2)

model_Num_Delinquent = lm(PRSM ~ (Num_Delinquent), data = installment2_id01)
plot(installment2_id01$Num_Delinquent, residuals(model_Num_Delinquent),
     xlab = "Num_Delinquent", ylab = "Residuals") +
  lines(lowess(installment2_id01$Num_Delinquent, residuals(model_Num_Delinquent))) +
  abline(h = 0, lty = 2)

model_Num_CreditLines = lm(PRSM ~ Num_CreditLines, data = installment2_id01)
plot(installment2_id01$Num_CreditLines, residuals(model_Num_CreditLines),
     xlab = "Num_CreditLines", ylab = "Residuals") +
  lines(lowess(installment2_id01$Num_CreditLines, residuals(model_Num_CreditLines))) +
  abline(h = 0, lty = 2)

model_Months = lm(PRSM ~ Months, data = installment2_id01)
plot(installment2_id01$Months, residuals(model_Months),
     xlab = "Months", ylab = "Residuals") +
  lines(lowess(installment2_id01$Months, residuals(model_Months))) +
  abline(h = 0, lty = 2)

# model_0: Original model ####

model_0 = lm(PRSM ~. , data = installment2_id01)
summary(model_0)

# FICOFair, FICOGood, FICoPoor, FICOVery Good, TotalAmtOwed, Stress, WomanOwned1,
# CorpStructureLLC, CorpStructurePartner, CorpStructureSole, NAICS445131, Months are statistically significant (ss)

# FICOFair, FICOGood, FICoPoor, FICOVery Good, Stress, WomanOwned1, 
# CorpStructureLLC, CorpStructurePartner, CorpStructureSole, NAICS445131, Months are discernible

## Cross Validation ####

cv10 = trainControl(method = "cv", number = 10)
model_0_cv = train(PRSM ~. , data = installment2_id01, method ="lm", trControl =cv10)
print(model_0_cv) # RMSE: 0.1031926       

# model_1: Model with p_delinquent ####

installment2_id01_1 = installment2_id01 %>% 
  mutate(p_delinquent = Num_Delinquent/Num_CreditLines)
# p_delinquent is not ss.

model_1 = lm(PRSM ~. - Num_Delinquent - Num_CreditLines, data = installment2_id01_1)
summary(model_1)

## Cross Validation ####

model_1_cv = train(PRSM ~. - Num_Delinquent - Num_CreditLines, data = installment2_id01_1, method ="lm", trControl =cv10)
print(model_1_cv) # RMSE: 0.1033768        

# model_2: Model with corpstructure & months interaction ####

model_2 = lm(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
summary(model_2)

## Cross Validation ####

model_2_cv = train(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
print(model_2_cv) # RMSE: 0.1029841           

# model_3: Model with raw FICO score ####

installment2_id01_2 = read.csv("installment2_id01.csv") %>%
  mutate(NAICS = as.factor(NAICS),
         WomanOwned = as.factor(WomanOwned),
         CorpStructure = as.factor(CorpStructure))
model_rm_outliers = lm(PRSM ~. , data = installment2_id01_2)
installment2_id01_2$residual=residuals(model_rm_outliers)

# remove 11 outliers observed in the plot (abs(residual)>=1)
installment2_id01_2 = installment2_id01_2 %>% 
  filter(abs(residual)<=1) %>% 
  select(-residual) 

# remove after certain months
model_rm_long_months = lm(PRSM ~ Months, data = installment2_id01_2)
plot(model_rm_long_months,1)
installment2_id01_2$fitted=fitted.values(model_rm_long_months)
installment2_id01_2 = installment2_id01_2 %>% 
  filter(abs(fitted)<=0.84) %>% 
  select(-fitted) 

model_3 = lm(PRSM ~. , data = installment2_id01_2)
summary(model_3)

## Cross Validation ####

model_3_cv = train(PRSM ~. , data = installment2_id01_2, method ="lm", trControl =cv10)
print(model_3_cv) # RMSE: 0.1063085     

# model_4: Model with corpstructure & months interaction + raw fico score ####

model_4 = lm(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS , data = installment2_id01_2)
summary(model_4)

## Cross Validation ####

model_4_cv = train(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS , data = installment2_id01_2, method ="lm", trControl =cv10)
print(model_4_cv) # RMSE: 0.1056763  

# model_5: Model with corpstructure & months interaction + transforming totalamtowed cube ####

model_5 = lm(PRSM ~ FICO + I(TotalAmtOwed^(1/2)) + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
summary(model_5)

## Cross Validation ####

model_5_cv = train(PRSM ~ FICO + I(TotalAmtOwed^(1/2)) + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
print(model_5_cv) # RMSE: 0.1071255      

# model_6: Model with corpstructure & months interaction + transforming totalamtowed & volume ####

model_6 = lm(PRSM ~ FICO + I(TotalAmtOwed^(0.5)) + I(Volume^(1/3)) + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
summary(model_6)

## Cross Validation ####

model_6_cv = train(PRSM ~ FICO + I(TotalAmtOwed^(0.5)) + I(Volume^(1/3)) + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
print(model_6_cv) # RMSE: 0.1067479      

# model_7: Model with corpstructure & months interaction + remove ####

model_7 = lm(PRSM ~ FICO + TotalAmtOwed + Stress + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
summary(model_7)

## Cross Validation ####

model_7_cv = train(PRSM ~ FICO + TotalAmtOwed + I(Stress^0.5) + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
print(model_7_cv) # RMSE: 0.1029772           

# model_8: Model with corpstructure & months interaction + transforming totalamtowed & removing volume and NAICS ####

model_8 = lm(PRSM ~ FICO + Stress + I(8*TotalAmtOwed) + WomanOwned + CorpStructure*Months, data = installment2_id01)
summary(model_8)

## Cross Validation ####

model_8_cv = train(PRSM ~ FICO + Stress + I(8*TotalAmtOwed) + WomanOwned + CorpStructure*Months, data = installment2_id01, method ="lm", trControl =cv10)
print(model_8_cv) # RMSE: 0.1028501           


# boxTidwell (transform)
?boxTidwell()
# Evaluation 1 

evaluation_data = read_csv(file = "installment2_evaluation_data.csv")
# here: do any necessary transformation or create any additional needed predictors
 
evaluation_data = evaluation_data %>%
  mutate(NAICS = as.factor(NAICS),
         WomanOwned = as.factor(WomanOwned),
         CorpStructure = as.factor(CorpStructure),
         FICO = case_when(
           (300 <= FICO)&(FICO <= 579) ~ "Poor",
           (580 <= FICO)&(FICO <= 669) ~ "Fair",
           (670 <= FICO)&(FICO <= 739) ~ "Good",
           (740 <= FICO)&(FICO <= 799) ~ "Very Good",
           (800 <= FICO)&(FICO <= 850) ~ "Excellent",
         ))

summary(installment2_id01$PRSM)
