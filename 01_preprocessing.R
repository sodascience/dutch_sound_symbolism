library(tidyverse)
library(readxl)

# Vankrunkelsven 2022 ----
vankrunkelsven_2022 <- read_delim(
  file = "raw_data/vankrunkelsven_2022/Norms_Gender.csv", 
  delim = ";", col_types = vankrunkelsven_2022_spec
)

write_rds(vankrunkelsven_2022, "processed_data/vankrunkelsven_2022.rds")

# brysbaert 2014 ----

