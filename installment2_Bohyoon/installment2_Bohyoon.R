rm(list = ls())

# install the environment 
library(tidyverse)

data = read.csv("installment2_id01.csv")
data = data %>%
  mutate(NAICS = as.character(NAICS),
         WomanOwned = as.character(WomanOwned),
         FICO = case_when(
           (300 <= FICO)&(FICO <= 579) ~ "Poor",
           (580 <= FICO)&(FICO <= 669) ~ "Fair",
           (670 <= FICO)&(FICO <= 739) ~ "Good",
           (740 <= FICO)&(FICO <= 799) ~ "Very Good",
           (800 <= FICO)&(FICO <= 850) ~ "Excellent",
         ))


model = lm(PRSM ~. , data = data)
summary(model)
