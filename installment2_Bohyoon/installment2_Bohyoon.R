rm(list = ls())

# install the environment 
library(tidyverse)

# read the data
data = read.csv("installment2_id01.csv")

# make NAICS, WomanOwned, FICO categorical variables
# make the chance of delinquency
data = data %>%
  mutate(NAICS = as.character(NAICS),
         WomanOwned = as.character(WomanOwned),
         FICO = case_when(
           (300 <= FICO)&(FICO <= 579) ~ "Poor",
           (580 <= FICO)&(FICO <= 669) ~ "Fair",
           (670 <= FICO)&(FICO <= 739) ~ "Good",
           (740 <= FICO)&(FICO <= 799) ~ "Very Good",
           (800 <= FICO)&(FICO <= 850) ~ "Excellent",
         ),
         p_delinquent = Num_Delinquent/Num_CreditLines)


model = lm(PRSM ~. , data = data)
summary(model)
