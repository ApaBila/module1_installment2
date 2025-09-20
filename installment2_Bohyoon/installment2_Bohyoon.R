rm(list = ls())

# install the environment 
library(tidyverse)

# read the data
installment2_id01 = read.csv("installment2_id01.csv")

# make NAICS, WomanOwned, FICO categorical variables
# add the chance of delinquency
installment2_id01 = installment2_id01 %>%
  mutate(NAICS = as.factor(NAICS),
         WomanOwned = as.factor(WomanOwned),
         FICO = case_when(
           (300 <= FICO)&(FICO <= 579) ~ "Poor",
           (580 <= FICO)&(FICO <= 669) ~ "Fair",
           (670 <= FICO)&(FICO <= 739) ~ "Good",
           (740 <= FICO)&(FICO <= 799) ~ "Very Good",
           (800 <= FICO)&(FICO <= 850) ~ "Excellent",
         ))


model = lm(PRSM ~. , data = installment2_id01)
installment2_id01$residual=residuals(model)

plot(model,1)

# remove 11 outliers observed in the plot (abs(residual)>=1)
installment2_id01_clean = installment2_id01 %>% 
  filter(abs(residual)<=1) %>% 
  select(-residual) 

model_clean = lm(PRSM ~. , data = installment2_id01_clean)

plot(model_clean,1)

summary(model_clean)

summary(installment2_id01_clean$PRSM)
hist(installment2_id01_clean$PRSM)

# 1. Model with p_delinquent ####

installment2_id01_clean_1 = installment2_id01_clean %>% 
  mutate(p_delinquent = Num_Delinquent/Num_CreditLines)
