# rm(list = ls())
# 
# # install the environment 
# library(tidyverse)
# library(caret)
# library(car)
# 
# # read the data
# installment2_id01 = read.csv("installment2_id01.csv")
# 
# # make NAICS, WomanOwned, FICO categorical variables
# 
# installment2_id01 = installment2_id01 %>%
#   mutate(NAICS = as.factor(NAICS),
#          WomanOwned = as.factor(WomanOwned),
#          CorpStructure = as.factor(CorpStructure),
#          FICO = case_when(
#            (300 <= FICO)&(FICO <= 579) ~ "Poor",
#            (580 <= FICO)&(FICO <= 669) ~ "Fair",
#            (670 <= FICO)&(FICO <= 739) ~ "Good",
#            (740 <= FICO)&(FICO <= 799) ~ "Very Good",
#            (800 <= FICO)&(FICO <= 850) ~ "Excellent",
#          ))
# 
# model_rm_outliers = lm(PRSM ~. , data = installment2_id01)
# installment2_id01$residual=residuals(model_rm_outliers)
# plot(model_rm_outliers,1)
# 
# # remove 11 outliers observed in the plot (abs(residual)>=1)
# installment2_id01 = installment2_id01 %>% 
#   filter(abs(residual)<=1) %>% 
#   dplyr::select(-residual) 
# 
# # TODO: remove after certain months
# 
# installment2_id01 = installment2_id01 %>% 
#   filter(abs(Months)<=60)
# 
# # numeric: TotalAmtOwed, Volume, Stress, Num_Delinquent, Num_CreditLines, Months
# 
# plot(installment2_id01$PRSM, installment2_id01$TotalAmtOwed); lines(lowess(installment2_id01$PRSM, installment2_id01$TotalAmtOwed))
# plot(installment2_id01$PRSM, installment2_id01$Volume)
# plot(installment2_id01$PRSM, installment2_id01$Stress)
# plot(installment2_id01$PRSM, installment2_id01$Num_Delinquent)
# plot(installment2_id01$PRSM, installment2_id01$Num_CreditLines)
# plot(installment2_id01$PRSM, installment2_id01$Months)
# 
# model_total = lm(PRSM ~ TotalAmtOwed, data = installment2_id01)
# plot(installment2_id01$TotalAmtOwed, residuals(model_total),
#      xlab = "TotalAmtOwed", ylab = "Residuals") +
# lines(lowess(installment2_id01$TotalAmtOwed, residuals(model_total))) +
# abline(h = 0, lty = 2)
# 
# model_volume = lm(PRSM ~ Volume, data = installment2_id01)
# plot(installment2_id01$Volume, residuals(model_volume),
#      xlab = "Volume", ylab = "Residuals") + 
#   lines(lowess(installment2_id01$Volume, residuals(model_volume))) +
#   abline(h = 0, lty = 2)
# 
# model_Stress = lm(PRSM ~ Stress, data = installment2_id01)
# plot(installment2_id01$Stress, residuals(model_Stress),
#      xlab = "Stress", ylab = "Residuals") +
# lines(lowess(installment2_id01$Stress, residuals(model_Stress))) +
# abline(h = 0, lty = 2)
# 
# model_Num_Delinquent = lm(PRSM ~ (Num_Delinquent), data = installment2_id01)
# plot(installment2_id01$Num_Delinquent, residuals(model_Num_Delinquent),
#      xlab = "Num_Delinquent", ylab = "Residuals") +
#   lines(lowess(installment2_id01$Num_Delinquent, residuals(model_Num_Delinquent))) +
#   abline(h = 0, lty = 2)
# 
# model_Num_CreditLines = lm(PRSM ~ Num_CreditLines, data = installment2_id01)
# plot(installment2_id01$Num_CreditLines, residuals(model_Num_CreditLines),
#      xlab = "Num_CreditLines", ylab = "Residuals") +
#   lines(lowess(installment2_id01$Num_CreditLines, residuals(model_Num_CreditLines))) +
#   abline(h = 0, lty = 2)
# 
# model_Months = lm(PRSM ~ Months, data = installment2_id01)
# plot(installment2_id01$Months, residuals(model_Months),
#      xlab = "Months", ylab = "Residuals") +
#   lines(lowess(installment2_id01$Months, residuals(model_Months))) +
#   abline(h = 0, lty = 2)
# 
# # model_0: Original model ####
# 
# model_0 = lm(PRSM ~. , data = installment2_id01)
# summary(model_0)
# 
# # FICOFair, FICOGood, FICoPoor, FICOVery Good, TotalAmtOwed, Stress, WomanOwned1,
# # CorpStructureLLC, CorpStructurePartner, CorpStructureSole, NAICS445131, Months are statistically significant (ss)
# 
# # FICOFair, FICOGood, FICoPoor, FICOVery Good, Stress, WomanOwned1, 
# # CorpStructureLLC, CorpStructurePartner, CorpStructureSole, NAICS445131, Months are discernible
# 
# ## Cross Validation ####
# 
# cv10 = trainControl(method = "cv", number = 10)
# model_0_cv = train(PRSM ~. , data = installment2_id01, method ="lm", trControl =cv10)
# print(model_0_cv) # RMSE: 0.1031926       
# 
# # model_1: Model with p_delinquent ####
# 
# installment2_id01_1 = installment2_id01 %>% 
#   mutate(p_delinquent = Num_Delinquent/Num_CreditLines)
# # p_delinquent is not ss.
# 
# model_1 = lm(PRSM ~. - Num_Delinquent - Num_CreditLines, data = installment2_id01_1)
# summary(model_1)
# 
# ## Cross Validation ####
# 
# model_1_cv = train(PRSM ~. - Num_Delinquent - Num_CreditLines, data = installment2_id01_1, method ="lm", trControl =cv10)
# print(model_1_cv) # RMSE: 0.1033768        
# 
# # model_2: Model with corpstructure & months interaction ####
# 
# model_2 = lm(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
# summary(model_2)
# 
# ## Cross Validation ####
# 
# model_2_cv = train(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
# print(model_2_cv) # RMSE: 0.1029841           
# 
# # model_3: Model with raw FICO score ####
# 
# installment2_id01_2 = read.csv("installment2_id01.csv") %>%
#   mutate(NAICS = as.factor(NAICS),
#          WomanOwned = as.factor(WomanOwned),
#          CorpStructure = as.factor(CorpStructure))
# model_rm_outliers = lm(PRSM ~. , data = installment2_id01_2)
# installment2_id01_2$residual=residuals(model_rm_outliers)
# 
# # remove 11 outliers observed in the plot (abs(residual)>=1)
# installment2_id01_2 = installment2_id01_2 %>% 
#   filter(abs(residual)<=1) %>% 
#   select(-residual) 
# 
# # remove after certain months
# model_rm_long_months = lm(PRSM ~ Months, data = installment2_id01_2)
# plot(model_rm_long_months,1)
# installment2_id01_2$fitted=fitted.values(model_rm_long_months)
# installment2_id01_2 = installment2_id01_2 %>% 
#   filter(abs(fitted)<=0.84) %>% 
#   select(-fitted) 
# 
# model_3 = lm(PRSM ~. , data = installment2_id01_2)
# summary(model_3)
# 
# ## Cross Validation ####
# 
# model_3_cv = train(PRSM ~. , data = installment2_id01_2, method ="lm", trControl =cv10)
# print(model_3_cv) # RMSE: 0.1063085     
# 
# # model_4: Model with corpstructure & months interaction + raw fico score ####
# 
# model_4 = lm(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS , data = installment2_id01_2)
# summary(model_4)
# 
# ## Cross Validation ####
# 
# model_4_cv = train(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS , data = installment2_id01_2, method ="lm", trControl =cv10)
# print(model_4_cv) # RMSE: 0.1056763  
# 
# # model_5: Model with corpstructure & months interaction + transforming totalamtowed cube ####
# 
# model_5 = lm(PRSM ~ FICO + I(TotalAmtOwed^(1/2)) + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
# summary(model_5)
# 
# ## Cross Validation ####
# 
# model_5_cv = train(PRSM ~ FICO + I(TotalAmtOwed^(1/2)) + Volume + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
# print(model_5_cv) # RMSE: 0.1071255      
# 
# # model_6: Model with corpstructure & months interaction + transforming totalamtowed & volume ####
# 
# model_6 = lm(PRSM ~ FICO + I(TotalAmtOwed^(0.5)) + I(Volume^(1/3)) + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
# summary(model_6)
# 
# ## Cross Validation ####
# 
# model_6_cv = train(PRSM ~ FICO + I(TotalAmtOwed^(0.5)) + I(Volume^(1/3)) + Stress + Num_Delinquent + Num_CreditLines + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
# print(model_6_cv) # RMSE: 0.1067479      
# 
# # model_7: Model with corpstructure & months interaction + remove ####
# 
# model_7 = lm(PRSM ~ FICO + TotalAmtOwed + Stress + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01)
# summary(model_7)
# 
# ## Cross Validation ####
# 
# model_7_cv = train(PRSM ~ FICO + TotalAmtOwed + I(Stress^0.5) + WomanOwned + CorpStructure*Months + NAICS, data = installment2_id01, method ="lm", trControl =cv10)
# print(model_7_cv) # RMSE: 0.1029772           
# 
# # model_8: Model with corpstructure & months interaction + transforming totalamtowed & removing volume and NAICS ####
# 
# model_8 = lm(PRSM ~ FICO + Stress + I(8*TotalAmtOwed) + WomanOwned + CorpStructure*Months, data = installment2_id01)
# summary(model_8)
# 
# ## Cross Validation ####
# 
# model_8_cv = train(PRSM ~ FICO + Stress + I(8*TotalAmtOwed) + WomanOwned + CorpStructure*Months, data = installment2_id01, method ="lm", trControl =cv10)
# print(model_8_cv) # RMSE: 0.1028501           
# 
# 
# # boxTidwell (transform)
# ?boxTidwell()
# # Evaluation 1 
# 
# evaluation_data = read_csv(file = "installment2_evaluation_data.csv")
# # here: do any necessary transformation or create any additional needed predictors
#  
# evaluation_data = evaluation_data %>%
#   mutate(NAICS = as.factor(NAICS),
#          WomanOwned = as.factor(WomanOwned),
#          CorpStructure = as.factor(CorpStructure),
#          FICO = case_when(
#            (300 <= FICO)&(FICO <= 579) ~ "Poor",
#            (580 <= FICO)&(FICO <= 669) ~ "Fair",
#            (670 <= FICO)&(FICO <= 739) ~ "Good",
#            (740 <= FICO)&(FICO <= 799) ~ "Very Good",
#            (800 <= FICO)&(FICO <= 850) ~ "Excellent",
#          ))
# 
# summary(installment2_id01$PRSM)


library(dplyr)
library(ModelMetrics)
library(MASS)
library(car)
library(caret)
library(glmnet)
library(ggplot2)

installment2_id01 <- read.csv("installment2_id01.csv")
str(installment2_id01)
summary(installment2_id01)

# make NAICS, WomanOwned, FICO categorical variables

installment2_id01 = installment2_id01 %>%
  mutate(NAICS = as.factor(substr(NAICS, 1, 6)), # TODO: talk about simplifying
         WomanOwned = as.factor(WomanOwned),
         FICO = case_when(
           (300 <= FICO)&(FICO <= 579) ~ "Poor",
           (580 <= FICO)&(FICO <= 669) ~ "Fair",
           (670 <= FICO)&(FICO <= 739) ~ "Good",
           (740 <= FICO)&(FICO <= 799) ~ "Very Good",
           (800 <= FICO)&(FICO <= 850) ~ "Excellent",
         ))


#We model PRSM with all predictors: FICO, Total Owed, Volume, Stress, Number of Delinquent Lines and Credit Lines, Woman Owned, Corporate Structure, NAICS, Months

fullmodel <- lm(PRSM ~., data = installment2_id01)
plot(fullmodel,1)
installment2_id01$residual=residuals(fullmodel)

# remove 11 outliers observed in the plot (abs(residual)>=1)
installment2_id01_clean = installment2_id01 %>% 
  filter(abs(residual)<=1) %>% 
  dplyr::select(-residual) # clean data (use this!!)

model_clean = lm(PRSM ~. , data = installment2_id01_clean) 
plot(model_clean,1) # these look much better! but perhaps discuss different outlier removal techniques

summary(model_clean)


# Months seems to need transformation
crPlots(model_clean) 

# this matches hint 1 of the assignment
crPlots(lm(PRSM ~ Months, data = installment2_id01_clean))

# Transform Months Model
boxTidwell(PRSM ~ TotalAmtOwed + Volume + Stress + Num_Delinquent + Num_CreditLines + Months,
           data = installment2_id01_clean)

transformed_model <- lm(PRSM ~ . - Months + I(Months^(-1.19487)),
                        data = installment2_id01_clean)

summary(transformed_model)
rmse(transformed_model$fitted.values, installment2_id01_clean$PRSM) # 0.1018195


# Step Model
step_model <- step(transformed_model, trace = 0, direction = "both")
summary(step_model)
# Step Model removed NAICS, Num_Delinquent, and Num_CreditLines
rmse(step_model$fitted.values, installment2_id01_clean$PRSM) # RMSE = 0.1025071


# 10-fold Cross-Validation
cv10 = trainControl(method = "cv", number = 10)
model_cv = train(PRSM ~ FICO + TotalAmtOwed + Volume + Stress + WomanOwned + 
                   CorpStructure + I(Months^(-1.19487)), data = installment2_id01_clean, 
                 method ="lm", trControl =cv10) # RMSE = 0.1031503
summary(model_cv)

# Volume is not significant (remove later)



# Find interactions using LASSO
installment2_id01_clean$Months_transformed =
  (installment2_id01_clean$Months)^(-1.19487)
x <- model.matrix(PRSM ~ .^2, 
                  data = installment2_id01_clean[,c("PRSM", "FICO",
                                                    "TotalAmtOwed","Stress",
                                                    "WomanOwned", "CorpStructure",
                                                    "Months_transformed")])[, -1]
y <- installment2_id01_clean$PRSM
cv_lasso_interactions <- cv.glmnet(x, y, alpha = 1, nfolds = 10)

best_lambda <- cv_lasso_interactions$lambda.min
lasso_coeffs <- coef(cv_lasso_interactions, s = best_lambda)

threshold <- 0.001
significant_terms <- lasso_coeffs[abs(lasso_coeffs[,1]) > threshold, , drop = FALSE]
print(significant_terms)

# potential interactions: FICO:Stress, FICO:WomanOwned, FICO:CorpStructure, FICO:Months_transformed, Stress:WomanOwned, Stress:CorpStructure, WomanOwned:CorpStructure, WomanOwned:Months_transformed, CorpStructure:Months_transformed

interact_model = lm(PRSM ~. + FICO:Stress + FICO:WomanOwned + FICO:CorpStructure 
                    + FICO:Months_transformed + Stress:WomanOwned 
                    + Stress:CorpStructure + WomanOwned:CorpStructure 
                    + WomanOwned:Months_transformed + CorpStructure:Months_transformed
                    , data = installment2_id01_clean[,c("PRSM", "FICO",
                                                        "TotalAmtOwed","Stress",
                                                        "WomanOwned", "CorpStructure",
                                                        "Months_transformed")])
summary(interact_model)
step_interact_model = step(interact_model, trace = 0, direction = "both")
summary(step_interact_model)
# Step Model removed NAICS, Num_Delinquent, and Num_CreditLines
rmse(step_interact_model$fitted.values, installment2_id01_clean$PRSM) # RMSE = 0.09830548