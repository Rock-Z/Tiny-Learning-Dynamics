library(ggplot2)
library(dplyr)
library(reshape2)

tiny_stories <- read.csv("/Users/Samuel/Desktop/SAMUEL MacBook/Samuel Yale/Courses/LING 380/Final Project/Visualizations/Eval Data/Tiny Blimp 1 Eval Data.csv")

more_stories <- read.csv("/Users/Samuel/Desktop/SAMUEL MacBook/Samuel Yale/Courses/LING 380/Final Project/Visualizations/Eval Data/Tiny Blimp 2 Eval Data.csv")

extra_stories_big <- read.csv("/Users/Samuel/Desktop/SAMUEL MacBook/Samuel Yale/Courses/LING 380/Final Project/Visualizations/Eval Data/Tiny Blimp 3 Eval Data Big.csv")


tiny_stories <- tiny_stories %>% filter(Checkpoint == 18500) %>% rbind(tiny_stories)
tiny_stories[1,1] <- 19000
tiny_stories <- arrange(tiny_stories, Checkpoint)

columns <- c('Checkpoint', 
             'Regular Plural Agreement 1', 'Regular Plural Agreement 2',
             'Anaphor Gender Agreement 1', 'Anaphor Number Agreement 2',
             'NPI Present 1', 'NPI Present 2',
             'Only NPI Licensor', 'Only NPI Scope',
             'Not NPI Licensing', 'Not NPI Scope',
             'Irregular Plural Agreement 1', 'Irregular Plural Agreement 2')

colnames(tiny_stories) <- columns
colnames(more_stories) <- columns
colnames(extra_stories_big) <- columns

tiny_stories <- tiny_stories %>% melt(id.vars = "Checkpoint", 
                                      value.name = 'Accuracy', 
                                      variable.name = 'Test Set')
more_stories <- more_stories %>% melt(id.vars = "Checkpoint", 
                                      value.name = 'Accuracy', 
                                      variable.name = 'Test Set')
extra_stories_big <- extra_stories_big %>% melt(id.vars = "Checkpoint", 
                                                value.name = 'Accuracy', 
                                                variable.name = 'Test Set')


joint <- merge(tiny_stories, more_stories, by=c('Checkpoint', 'Test Set')) %>% 
  melt(id.vars = c('Checkpoint', 'Test Set'), variable.name = 'Model', value.name = 'Accuracy') %>% 
  arrange(Checkpoint, `Test Set`)

joint %>% ggplot(aes(Checkpoint, Accuracy, group = Model, color = Model)) +
  ggtitle('Evaluation, Global') +
  geom_line() +
  facet_wrap(vars(`Test Set`), ncol = 3)

Earlymeans <- joint %>% filter(Checkpoint <= 2000) %>% group_by(`Test Set`, Model) %>% summarise(mean = mean(Accuracy))
Latemeans <- joint %>% filter(Checkpoint >= 47500) %>% group_by(`Test Set`, Model) %>% summarise(mean = mean(Accuracy))

Earlymeans
Latemeans[4] <- Latemeans[3] - Earlymeans[3]

Latemeans %>% select(c(1, 2, 4)) %>%
  ggplot(aes(`Test Set`, mean, color = Model)) +
  geom_point(size=5)

joint %>%
  arrange(`Test Set`) %>% filter(Model == 'Accuracy.y') %>% group_by(`Test Set`) %>% 
  mutate(lagval = lag(Accuracy, order_by=Checkpoint)) %>%
  tidyr::fill(`lagval`, .direction = "up") %>%
  mutate(increase = (Accuracy * 100 / lagval) - 100) %>%
  ggplot(aes(Checkpoint, increase, color = Model, ylim(-50, 70))) +
  ggtitle('Percentage change in accuracy') +
  geom_line() +
  facet_wrap(vars(`Test Set`), ncol = 3)

joint %>% filter(Model == 'Accuracy.y' & `Test Set` == 'Irregular Plural Agreement 2') %>%
  ggplot(aes(Checkpoint, Accuracy)) +
  geom_line(color = 'skyblue') +
  ggtitle('Irregular Plural Agreement 2')