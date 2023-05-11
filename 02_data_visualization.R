library('tidyverse')

brysbaert_2014 <- read_rds('processed_data/brysbaert_2014.rds')
speed_2021 <- read_rds('processed_data/speed_2021.rds')
vankrunkelsven_2022 <- read_rds('processed_data/vankrunkelsven_2022.rds')

# Brysbaert ----
## Concreteness 
concreteness_hist <- hist(brysbaert_2014$concreteness_brysbaert)
concreteness_summary <- summary(brysbaert_2014$concreteness_brysbaert)
concreteness_summary



# Speed ----
## Auditory

auditory_hist <- hist(speed_2021$auditory_speedb)
auditory_summary <- summary(speed_2021$auditory_speedb)
auditory_summary

## Visual

visual_hist <- hist(speed_2021$visual_speedb)
visual_summary <- summary(speed_2021$visual_speedb)
visual_summary

## Olfactory

olfactory_hist <- hist(speed_2021$olfactory_speedb)
olfactory_summary <- summary(speed_2021$olfactory_speedb)
olfactory_summary

## Gustatory

gustatory_hist <- hist(speed_2021$gustatory_speedb)
gustatory_summary <- summary(speed_2021$gustatory_speedb)
gustatory_summary

## Haptic

haptic_hist <- hist(speed_2021$haptic_speedb)
haptic_summary <- summary(speed_2021$haptic_speedb)
haptic_summary

## Interoceptive

interoceptive_hist <- hist(speed_2021$interoceptive_speedb)
interoceptive_summary <- summary(speed_2021$interoceptive_speedb)
interoceptive_summary


# Vankrunkelsven ----
## Gender 

gender_hist <- hist(vankrunkelsven_2022$gender_vankrunkelsven)
gender_summary <- summary(vankrunkelsven_2022$gender_vankrunkelsven)
gender_summary



