library(tidyverse)
library(readxl)

# Vankrunkelsven 2022 ----
vankrunkelsven_2022 <- read_delim(
  file = "raw_data/vankrunkelsven_2022/Norms_Gender.csv", 
  delim = ";"
) %>%
  select(Word, Gender) %>%
  rename(word = Word,
         gender_vankrunkelsven = Gender)

# write file to disk
write_rds(vankrunkelsven_2022, "processed_data/vankrunkelsven_2022.rds")

# Brysbaert 2014 ----
brysbaert_2014 <- read_xlsx(
  "raw_data/brysbaert_2014/1-s2.0-S0001691814000985-mmc3.xlsx"
) %>%
  select(stimulus, Concrete_m) %>%
  rename(word = stimulus,
         concreteness_brysbaert = Concrete_m)

# write file to disk
write_rds(brysbaert_2014, "processed_data/brysbaert_2014.rds")

# Speed 2021 ----
speed_2021 <- read_xlsx(
  "raw_data/speed_2021/SpeedBrysbaert_Norms.xlsx"
) %>%
  select(Woord, Horen, Zien, Ruiken, Proeven, Voelen, Sensaties) %>%
  rename(word = Woord,
         auditory_speedb = Horen,
         visual_speedb = Zien,
         olfactory_speedb = Ruiken,
         gustatory_speedb = Proeven,
         haptic_speedb = Voelen,
         interoceptive_speedb = Sensaties)

# write file to disk 
write_rds(speed_2021, "processed_data/speed_2021.rds")
