library(ggplot2)
library(mgcv)
library(itsadug)
library(lme4)
library(lmerTest)
library(ggeffects)
library(MASS)
library(plyr)
library(dplyr)
library(tidyr)
library(tidyverse)
library(ggbeeswarm)

data_path <- './processed_data/analyses/regression_analysis/'

company_names <- read.csv('./processed_data/words_and_names/survey_lists/company_names_final.csv')
colnames(company_names) <- c('name')
company_names['word_type'] <- 'company_name'
first_names <- read.csv('./processed_data/words_and_names/survey_lists/dutch_names_final.csv')
colnames(first_names) <- c('name')
first_names['word_type'] <- 'first_name'
nonwords <- read.csv('./processed_data/words_and_names/survey_lists/nonwords_final.csv')
colnames(nonwords) <- c('name')
nonwords['word_type'] <- 'nonword'
names <- rbind(company_names,first_names, nonwords)
names <- mutate_all(names, .funs=tolower)

file_names <- list.files(data_path, include.dirs = TRUE)

df <- data.frame(matrix(ncol = 7, nrow = 0))
colnames(df) <- c('name', 'feature_vector', 'true_rating', 'predicted_rating', 
                  'emb_type', 'emb_model', 'association')

for (file in file_names){
  temp_df <- read_csv(paste0(data_path, file), col_select = 2:5)

  
  embtype <- gsub('[^[:alpha:]]', '', strsplit(file, '_')[[1]][2])
  embmodel <- gsub('[[:alpha:]]', '', strsplit(file, '_')[[1]][2])
  association <- strsplit(file, '_')[[1]][1]
  
  temp_df['emb_type'] <- embWordType
  temp_df['emb_model'] <- embmodel
  temp_df['association'] <- association
  
  df <- rbind(df, temp_df)
  
}

df_processed <- df %>%
  mutate(emb_type=as.factor(emb_type)) %>%
  mutate(emb_model=as.factor(emb_model)) %>%
  mutate(association=as.factor(association))

df_processed <- merge(df_processed, names, by = 'name') %>%
  mutate(word_type=as.factor(word_type))

#### PREDICTIONS #####
## FEMININE

df.fem <- df_processed[df_processed$association == 'feminine', ]

lm.fem.base <- lm(true_rating ~ word_type, data = df.fem)
lm.fem.preds <- lm(true_rating ~ word_type + predicted_rating + emb_type, data = df.fem)
lm.fem.preds.int <- lm(true_rating ~ word_type*predicted_rating*emb_type, data = df.fem)

AIC(lm.fem.base) # 4330.900
AIC(lm.fem.preds) # 2071.489
AIC(lm.fem.preds.int) # 1911.766, 4336.900 for wtype*emb_type, 2067.979 for emb_type*rating, 2017.941 for wtype*rating

summary(lm.fem.base)
summary(lm.fem.preds) 
summary(lm.fem.preds.int)

fem.preds = data.frame(
  ggpredict(lm.fem.preds.int, terms = c(
    "predicted_rating",
    "emb_type",
    "word_type")
  )
)
names(fem.preds)[names(fem.preds) == 'x'] <- 'modelPrediction'
names(fem.preds)[names(fem.preds) == 'group'] <- 'emb_type'
names(fem.preds)[names(fem.preds) == 'facet'] <- 'word_type'
names(fem.preds)[names(fem.preds) == 'predicted'] <- 'Rating'

pdf(file = "plots/predictions/lm.fem.pdf",  
    width = 7,
    height = 5)
ggplot(data = fem.preds, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = emb_type,  linetype = emb_type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = emb_type, fill=emb_type), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color = emb_type), size=0.5, data=df.fem, show.legend = FALSE) +
  facet_grid(. ~ word_type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with fem rating, by emb_type and name word_type') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_linetype_manual(values = c(
    "bert" = "dashed", 
    "ft" = "solid")) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()

## GOOD

df.good <- df_processed[df_processed$association == 'good', ]

lm.good.base <- lm(true_rating ~ word_type, data = df.good)
lm.good.preds <- lm(true_rating ~ word_type + predicted_rating + emb_type, data = df.good)
lm.good.preds.int <- lm(true_rating ~ word_type*predicted_rating*emb_type, data = df.good)

AIC(lm.good.base) # 1440.503
AIC(lm.good.preds) # 581.016
AIC(lm.good.preds.int) # 555.9429, 1446.503 for wtype*emb_type, 670.5158 for emb_type*rating, 575.8993 for wtype*rating

summary(lm.good.base)
summary(lm.good.preds) 
summary(lm.good.preds.int)

good.preds = data.frame(
  ggpredict(lm.good.preds.int, terms = c(
    "predicted_rating",
    "emb_type",
    "word_type")
  )
)
names(good.preds)[names(good.preds) == 'x'] <- 'modelPrediction'
names(good.preds)[names(good.preds) == 'group'] <- 'emb_type'
names(good.preds)[names(good.preds) == 'facet'] <- 'word_type'
names(good.preds)[names(good.preds) == 'predicted'] <- 'Rating'

pdf(file = "plots/predictions/lm.good.pdf",  
    width = 7,
    height = 5)
ggplot(data = good.preds, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = emb_type,  linetype = emb_type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = emb_type, fill=emb_type), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color = emb_type), size=0.5, data=df.good, show.legend = FALSE) +
  facet_grid(. ~ word_type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with good rating, by emb_type and name word_type') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_linetype_manual(values = c(
    "bert" = "dashed", 
    "ft" = "solid")) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()

## SMART

df.smart <- df_processed[df_processed$association == 'smart', ]

lm.smart.base <- lm(true_rating ~ word_type, data = df.smart)
lm.smart.preds <- lm(true_rating ~ word_type + predicted_rating + emb_type, data = df.smart)
lm.smart.preds.int <- lm(true_rating ~ word_type*predicted_rating*emb_type, data = df.smart)

AIC(lm.smart.base) # 1077.477
AIC(lm.smart.preds) # 161.3406
AIC(lm.smart.preds.int) # 166.0221, 1083.477 for wtype*emb_type, 166.6233 for emb_type*rating, 162.4978 for wtype*rating

summary(lm.smart.base)
summary(lm.smart.preds) 
summary(lm.smart.preds.int)

smart.preds = data.frame(
  ggpredict(lm.smart.preds.int, terms = c(
    "predicted_rating",
    "emb_type",
    "word_type")
  )
)
names(smart.preds)[names(smart.preds) == 'x'] <- 'modelPrediction'
names(smart.preds)[names(smart.preds) == 'group'] <- 'emb_type'
names(smart.preds)[names(smart.preds) == 'facet'] <- 'word_type'
names(smart.preds)[names(smart.preds) == 'predicted'] <- 'Rating'

pdf(file = "plots/predictions/lm.smart.pdf",  
    width = 7,
    height = 5)
ggplot(data = smart.preds, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = emb_type,  linetype = emb_type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = emb_type, fill=emb_type), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color = emb_type), size=0.5, data=df.smart, show.legend = FALSE) +
  facet_grid(. ~ word_type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with smart rating, by emb_type and name word_type') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_linetype_manual(values = c(
    "bert" = "dashed", 
    "ft" = "solid")) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()

## TRUSTWORTHY

df.trust <- df_processed[df_processed$association == 'trustworthy', ]

lm.trust.base <- lm(true_rating ~ word_type, data = df.trust)
lm.trust.preds <- lm(true_rating ~ word_type + predicted_rating + emb_type, data = df.trust)
lm.trust.preds.int <- lm(true_rating ~ word_type*predicted_rating*emb_type, data = df.trust)

AIC(lm.trust.base) # 891.7017
AIC(lm.trust.preds) # 59.33164
AIC(lm.trust.preds.int) # 31.90096, 897.7017 for wtype*emb_type, 157.9907 for emb_type*rating, 24.36337 for wtype*rating

summary(lm.trust.base)
summary(lm.trust.preds) 
summary(lm.trust.preds.int)

trust.preds = data.frame(
  ggpredict(lm.trust.preds.int, terms = c(
    "predicted_rating",
    "emb_type",
    "word_type")
  )
)
names(trust.preds)[names(trust.preds) == 'x'] <- 'modelPrediction'
names(trust.preds)[names(trust.preds) == 'group'] <- 'emb_type'
names(trust.preds)[names(trust.preds) == 'facet'] <- 'word_type'
names(trust.preds)[names(trust.preds) == 'predicted'] <- 'Rating'

pdf(file = "plots/predictions/lm.trust.pdf",  
    width = 7,
    height = 5)
ggplot(data = trust.preds, aes(x = modelPrediction, y = Rating)) +
  geom_line(aes(color = emb_type,  linetype = emb_type)) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, color = emb_type, fill=emb_type), alpha = 0.4) +
  geom_point(aes(x=predicted_rating, y=true_rating, color = emb_type), size=0.5, data=df.trust, show.legend = FALSE) +
  facet_grid(. ~ word_type) +
  ylim(-1, 1) +
  ggtitle('Prediction relation with trust rating, by emb_type and name word_type') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) + scale_linetype_manual(values = c(
    "bert" = "dashed", 
    "ft" = "solid")) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'vertical', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) 
dev.off()







#### ERRORS ####
## FEMININE
df.fem = df_processed[df_processed$association == 'feminine', ]

df.fem = df.fem %>%
  group_by(word_type) %>%
  mutate(base_error = true_rating - mean(true_rating),
         absolute_error = abs(true_rating - predicted_rating),
         absolute_base_error = abs(base_error))

df.fem$err_diff = abs(df.fem$absolute_base_error - df.fem$absolute_error)

pdf(file = "plots/MAE/violin.fem.pdf",   
    width = 12, 
    height = 7)
ggplot(data = df.fem, aes(x = emb_type, y=absolute_error)) +
  geom_violin(aes(color = emb_type, fill=emb_type), position = position_dodge(width = 0.7)) +
  geom_quasirandom(aes(color = emb_type), size=0.25, dodge.width = 0.7) +
  ggtitle('Absolute error, by embedding type - feminine') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  facet_grid(word_type ~ emb_model) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) +
  guides(alpha = guide_legend(override.aes = list(fill = c('darkgrey','darkgrey'))))
dev.off()



lm.fem.err.base = lm(absolute_error ~ 1, 
                        data = df.fem)
AIC(lm.fem.err.base)  # 709.6265
lm.fem.err.type = lm(absolute_error ~ word_type, 
                        data = df.fem)
AIC(lm.fem.err.type)  # 648.4716
lm.fem.err.feat = lm(absolute_error ~ emb_type, 
                        data = df.fem)
AIC(lm.fem.err.feat)  # 653.5329
lm.fem.err.comb = lm(absolute_error ~ word_type + emb_type, 
                        data = df.fem)
AIC(lm.fem.err.comb)  # 590.2005
lm.fem.err.int = lm(absolute_error ~ word_type*emb_type, 
                       data = df.fem)
AIC(lm.fem.err.int)  # 536.3572
summary(lm.fem.err.int)


fem.preds.err = data.frame(
  ggpredict(lm.fem.err.int, terms = c(
    "word_type",
    "emb_type"
  )
  )
)
names(fem.preds.err)[names(fem.preds.err) == 'x'] <- 'type'
names(fem.preds.err)[names(fem.preds.err) == 'group'] <- 'emb_type'
names(fem.preds.err)[names(fem.preds.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.fem.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = fem.pred.err, aes(x=emb_type, y=AbsoluteError)) +
  geom_bar(aes(fill=emb_type, color=emb_type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=emb_type, y=absolute_error, color=emb_type), size=0.5, data=df.fem, position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = emb_type), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  facet_grid(cols = vars(word_type)) +
  ggtitle('MAE (ANN predictions) - feminine') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x=element_blank()
  )
dev.off()

## GOOD
df.good = df_processed[df_processed$association == 'good', ]

df.good = df.good %>%
  group_by(word_type) %>%
  mutate(base_error = true_rating - mean(true_rating),
         absolute_error = abs(true_rating - predicted_rating),
         absolute_base_error = abs(base_error))

df.good$err_diff = abs(df.good$absolute_base_error - df.good$absolute_error)

pdf(file = "plots/MAE/violin.good.pdf",   
    width = 12, 
    height = 7)
ggplot(data = df.good, aes(x = emb_type, y=absolute_error)) +
  geom_violin(aes(color = emb_type, fill=emb_type), position = position_dodge(width = 0.7)) +
  geom_quasirandom(aes(color = emb_type), size=0.25, dodge.width = 0.7) +
  ggtitle('Absolute error, by embedding type - good') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  facet_grid(word_type ~ emb_model) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) +
  guides(alpha = guide_legend(override.aes = list(fill = c('darkgrey','darkgrey'))))
dev.off()



lm.good.err.base = lm(absolute_error ~ 1, 
                     data = df.good)
AIC(lm.good.err.base)  # -1116.305
lm.good.err.type = lm(absolute_error ~ word_type, 
                     data = df.good)
AIC(lm.good.err.type)  # -1156.999
lm.good.err.feat = lm(absolute_error ~ emb_type, 
                     data = df.good)
AIC(lm.good.err.feat)  # -1114.312
lm.good.err.comb = lm(absolute_error ~ word_type + emb_type, 
                     data = df.good)
AIC(lm.good.err.comb)  # -1155.007
lm.good.err.int = lm(absolute_error ~ word_type*emb_type, 
                    data = df.good)
AIC(lm.good.err.int)  # -1156.835
summary(lm.good.err.int)


good.preds.err = data.frame(
  ggpredict(lm.good.err.int, terms = c(
    "word_type",
    "emb_type"
  )
  )
)
names(good.preds.err)[names(good.preds.err) == 'x'] <- 'type'
names(good.preds.err)[names(good.preds.err) == 'group'] <- 'emb_type'
names(good.preds.err)[names(good.preds.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.good.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = good.preds.err, aes(x=emb_type, y=AbsoluteError)) +
  geom_bar(aes(fill=emb_type, color=emb_type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=emb_type, y=absolute_error, color=emb_type), size=0.5, data=df.good, position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = emb_type), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  facet_grid(cols = vars(word_type)) +
  ggtitle('MAE (ANN predictions) - good') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x=element_blank()
  )
dev.off()


## SMART
df.smart = df_processed[df_processed$association == 'smart', ]

df.smart = df.smart %>%
  group_by(word_type) %>%
  mutate(base_error = true_rating - mean(true_rating),
         absolute_error = abs(true_rating - predicted_rating),
         absolute_base_error = abs(base_error))

df.smart$err_diff = abs(df.smart$absolute_base_error - df.smart$absolute_error)

pdf(file = "plots/MAE/violin.smart.pdf",   
    width = 12, 
    height = 7)
ggplot(data = df.smart, aes(x = emb_type, y=absolute_error)) +
  geom_violin(aes(color = emb_type, fill=emb_type), position = position_dodge(width = 0.7)) +
  geom_quasirandom(aes(color = emb_type), size=0.25, dodge.width = 0.7) +
  ggtitle('Absolute error, by embedding type - smart') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  facet_grid(word_type ~ emb_model) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) +
  guides(alpha = guide_legend(override.aes = list(fill = c('darkgrey','darkgrey'))))
dev.off()


lm.smart.err.base = lm(absolute_error ~ 1, 
                     data = df.smart)
AIC(lm.smart.err.base)  # -1448.999
lm.smart.err.type = lm(absolute_error ~ word_type, 
                     data = df.smart)
AIC(lm.smart.err.type)  # -1467.158
lm.smart.err.feat = lm(absolute_error ~ emb_type, 
                     data = df.smart)
AIC(lm.smart.err.feat)  # -1503.15
lm.smart.err.comb = lm(absolute_error ~ word_type + emb_type, 
                     data = df.smart)
AIC(lm.smart.err.comb)  # -1522.015
lm.smart.err.int = lm(absolute_error ~ word_type*emb_type, 
                    data = df.smart)
AIC(lm.smart.err.int)  # -1524.66
summary(lm.smart.err.int)


smart.preds.err = data.frame(
  ggpredict(lm.smart.err.int, terms = c(
    "word_type",
    "emb_type"
  )
  )
)
names(smart.preds.err)[names(smart.preds.err) == 'x'] <- 'type'
names(smart.preds.err)[names(smart.preds.err) == 'group'] <- 'emb_type'
names(smart.preds.err)[names(smart.preds.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.smart.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = smart.preds.err, aes(x=emb_type, y=AbsoluteError)) +
  geom_bar(aes(fill=emb_type, color=emb_type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=emb_type, y=absolute_error, color=emb_type), size=0.5, data=df.smart, position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = emb_type), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  facet_grid(cols = vars(word_type)) +
  ggtitle('MAE (ANN predictions) - smart') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x=element_blank()
  )
dev.off()


## TRUSTWORTHY
df.trust = df_processed[df_processed$association == 'trustworthy', ]

df.trust = df.trust %>%
  group_by(word_type) %>%
  mutate(base_error = true_rating - mean(true_rating),
         absolute_error = abs(true_rating - predicted_rating),
         absolute_base_error = abs(base_error))

df.trust$err_diff = abs(df.trust$absolute_base_error - df.trust$absolute_error)

pdf(file = "plots/MAE/violin.trust.pdf",   
    width = 12, 
    height = 7)
ggplot(data = df.trust, aes(x = emb_type, y=absolute_error)) +
  geom_violin(aes(color = emb_type, fill=emb_type), position = position_dodge(width = 0.7)) +
  geom_quasirandom(aes(color = emb_type), size=0.25, dodge.width = 0.7) +
  ggtitle('Absolute error, by embedding type - trustworthy') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  facet_grid(word_type ~ emb_model) +
  theme(
    legend.justification = 'right', 
    legend.position = 'bottom', 
    legend.box = 'horizontal', 
    legend.box.just = 'right',
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    axis.title.x=element_blank()
  ) +
  guides(alpha = guide_legend(override.aes = list(fill = c('darkgrey','darkgrey'))))
dev.off()


lm.trust.err.base = lm(absolute_error ~ 1, 
                     data = df.trust)
AIC(lm.trust.err.base)  # -1450.506
lm.trust.err.type = lm(absolute_error ~ word_type, 
                     data = df.trust)
AIC(lm.trust.err.type)  # -1504.852
lm.trust.err.feat = lm(absolute_error ~ emb_type, 
                     data = df.trust)
AIC(lm.trust.err.feat)  # -1454.08
lm.trust.err.comb = lm(absolute_error ~ word_type + emb_type, 
                     data = df.trust)
AIC(lm.trust.err.comb)  # -1508.609
lm.trust.err.int = lm(absolute_error ~ word_type*emb_type, 
                    data = df.trust)
AIC(lm.trust.err.int)  # -1509.055
summary(lm.trust.err.int)


trust.preds.err = data.frame(
  ggpredict(lm.trust.err.int, terms = c(
    "word_type",
    "emb_type"
  )
  )
)
names(trust.preds.err)[names(trust.preds.err) == 'x'] <- 'type'
names(trust.preds.err)[names(trust.preds.err) == 'group'] <- 'emb_type'
names(trust.preds.err)[names(trust.preds.err) == 'predicted'] <- 'AbsoluteError'

pdf(file = "plots/MAE/bars.trust.lm.pdf",   # The directory you want to save the file in
    width = 7, # The width of the plot in inches
    height = 5)
ggplot(data = trust.preds.err, aes(x=emb_type, y=AbsoluteError)) +
  geom_bar(aes(fill=emb_type, color=emb_type), stat = 'identity', width = .5, position = 'dodge', alpha = 0.5) +
  geom_point(aes(x=emb_type, y=absolute_error, color=emb_type), size=0.5, data=df.trust, position=position_dodge(0.5)) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high, group = emb_type), 
                width = 0.2, color = 'slategrey', position = position_dodge(0.5)) +
  facet_grid(cols = vars(word_type)) +
  ggtitle('MAE (ANN predictions) - trustworthy') +
  scale_fill_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  scale_color_manual(values = c(
    "bert" = "mediumorchid4",
    "ft" = "deeppink1"
  )) +
  theme(
    legend.justification = c(1, 0),
    legend.position = "bottom",
    legend.direction = "horizontal",
    strip.text.x = element_text(size = 10),
    strip.text.y = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x=element_blank()
  )
dev.off()

