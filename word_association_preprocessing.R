library('haven')
library('tidyverse')
library('readxl')

main_path = 'C:/Users/jooss004/Documents/Main Traineeship Project/Word Associations/'

bernabeau_data = read.csv(paste(main_path, 'Bernabeau, 2018/all.csv', sep = ''))
brysbaert_data = read_excel(paste(main_path, 'Brysbaert et al., 2014/data.xlsx', sep = ''))
deyne_data = read.csv2(paste(main_path, 'De Deyne et al., 2008/imageability_all_samengevoegd.csv', sep = ''), dec = '.')
hermans_data = read_csv2(paste(main_path, 'Hermans & De Houwer, 1994/data_handcopied.csv', sep = ''))
moors_data = read_excel(paste(main_path, 'Moors et al., 2013/data_precleaned.xlsx', sep =''))
roest_data = read.csv2(paste(main_path, 'Roest et al., 2018/DTN overall CSV.csv', sep =''), dec = '.')
speed_brysbaert_data = read_excel(paste(main_path, 'Speed & Brybaert., 2021/SpeedBrysbaert_Norms.xlsx', sep=''), sheet = 'SensoryRatings')
speed_majid_data = read_excel(paste(main_path, 'Speed & Majid, 2017/data_precleaned.xlsx', sep =''), sheet = 'AllRatings')
spooren_data = read_sav(paste(main_path, 'Spooren et al., 2015/Totaal_breed.sav', sep = ''))
vandergoten_data = read.csv2(paste(main_path, 'Van der Goten et al., 1999/data_handcopied.csv', sep = ''), dec = '.')
vankrunkelsven_data = read.csv2(paste(main_path, 'Vankrunkelsven et al., 2022/Norms_Gender.csv', sep = ''), dec = '.')
verheyen_data = read_excel(paste(main_path, 'Verheyen et al., 2020/data_precleaned.xlsx', sep = ''), sheet = 'AllData')
verkes_data = read_tsv(paste(main_path, 'Verkes et al., 1989/data_handcopied.csv', sep = ''))
vanloon_data = read_csv2(paste(main_path, 'van Loon-Vervoorn, 1985/data_handcopied.csv', sep = ''))


#### CLEANING DATAFRAMES #######################################################

bernabeau_data <- bernabeau_data %>%
  select(word, Auditory, Haptic, Visual) %>%
  rename(auditory_bernabeau = Auditory, 
         haptic_bernabeau = Haptic, 
         visual_bernabeau = Visual)

brysbaert_data <- brysbaert_data %>%
  select(stimulus, Concrete_m) %>%
  rename(word = stimulus,
         concreteness_brysbaert = Concrete_m)

deyne_data <- rename(deyne_data, imageability_deyne = imageability)

hermans_data <- rename(hermans_data, word = Word,
                       affectivity_hermans = Affective,
                       familiarity_hermans = Familiarity)

moors_data <- moors_data %>%
  select(Words, "M V", "M A", "M P") %>%
  rename(word = Words,
         valence_moors = "M V",
         arousal_moors = "M A",
         dominance_moors = "M P")

roest_data <- roest_data %>%
  select(Words, Arousal_M, Valence_M, Taboo_general_M, Taboo_personal_M, Insulting_M) %>%
  rename(word = Words,
         arousal_roest = Arousal_M, 
         valence_roest = Valence_M,
         taboo_general_roest = Taboo_general_M,
         taboo_personal_roest = Taboo_personal_M,
         insulting_roest = Insulting_M)

speed_brysbaert_data <- speed_brysbaert_data %>%
  select(Woord, Horen, Zien, Ruiken, Proeven, Voelen, Sensaties) %>%
  rename(word = Woord,
         auditory_speedb = Horen,
         visual_speedb = Zien,
         olfactory_speedb = Ruiken,
         gustatory_speedb = Proeven,
         haptic_speedb = Voelen,
         interoceptive_speedb = Sensaties)

speed_majid_data <- speed_majid_data %>%
  select(Word, "te horen", "te proeven", "te ruiken", "te voelen door middel van aanraking", 
         "te zien", Arousal, Dominance, Valence) %>%
  rename(word = Word,
         auditory_speedm = "te horen",
         gustatory_speedm = 'te proeven', 
         olfactory_speedm = 'te ruiken',
         haptic_speedm = 'te voelen door middel van aanraking',
         visual_speedm = 'te zien',
         arousal_speedm = Arousal,
         dominance_speedm = Dominance,
         valence_speedm = Valence)
speed_majid_data$word <- tolower(speed_majid_data$word)

for (row in 1:nrow(spooren_data)){
  spooren_data$stimulus[row] = strsplit(spooren_data$stimulus[row], ' ')[[1]][1]
}
spooren_data <- spooren_data %>%
  select(-diff) %>%
  rename(word = stimulus,
         concreteness_spooren = concreet,
         specificness_spooren = specifiek,
         sensability_spooren = zintwaarneem,
         understandability_spooren = begrijpelijk,
         drawfilmability_spooren = tekenfilmbaar)

vandergoten_data <- rename(vandergoten_data, word = Word,
                           concreteness_goten = Concreteness,
                           valence_emotional_goten = Emotional_Valence)

vankrunkelsven_data <- vankrunkelsven_data  %>%
  select(Word, Gender) %>%
  rename(word = Word,
         gender_krunkelsven = Gender)

vanloon_data <- rename(vanloon_data, imageability_loon = imageability)

verheyen_data <- rename(verheyen_data, word = Words,
                        arousal_verheyen = Arousal,
                        concreteness_verheyen = Concreteness,
                        dominance_verheyen = Dominance,
                        familiarity_verheyen = Familiarity,
                        imageability_verheyen = Imageability,
                        valence_verheyen = Valence)

verkes_data <- rename(verkes_data, pain_intensity_verkes = pain_intensity)

#### COMBINING ALL (EXCEPT VANLOON ATM) DATAFRAMES INTO ONE ####################

association_data <- full_join(brysbaert_data, bernabeau_data) %>%
  full_join(deyne_data) %>%
  full_join(hermans_data) %>%
  full_join(moors_data) %>%
  full_join(roest_data) %>%
  full_join(speed_brysbaert_data) %>%
  full_join(speed_majid_data) %>%
  full_join(spooren_data) %>%
  full_join(vandergoten_data) %>%
  full_join(vankrunkelsven_data) %>%
# full_join(vanloon_data) %>%
  full_join(verheyen_data) %>%
  full_join(verkes_data) %>%
  arrange(word)

association_data <- select(association_data, order(colnames(association_data)))

association_data <- relocate(association_data, word, .before = affectivity_hermans)

#### DESCRIPTIVE STATISTICS ####################################################

summary(association_data)

for (column in 2:ncol(association_data)){
  print(c(colnames(association_data[, column]),
        skewness(association_data[, column], na.rm = TRUE)))
  hist(unlist(association_data[colnames(association_data[, column])]))
  remove(column)
} 

  