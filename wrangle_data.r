library(tidyverse) 

data <- read.csv("indian_liver_patient.csv", header=TRUE, 
                 na.strings = c("NA","na",""," ","?"), stringsAsFactors = FALSE)

save(data, file = "rdas/data.rda")
