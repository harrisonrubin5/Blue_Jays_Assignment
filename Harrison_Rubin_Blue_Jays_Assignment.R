#### Toronto Blue Jays Baseball Research Analyst Assignment -- Harrison Rubin

# Load Packages and Set Seed
library(tidyverse)
library(sqldf)
library(tidymodels)
library(janitor)
library(naniar)
library(vip)
library(lsr)
library(stacks)
library(NeuralNetTools)
library(earth)
set.seed(8675309)

## Read in Data
training <- read_csv("training.csv") %>% 
  mutate(SpinRate = as.numeric(SpinRate)) %>% 
  mutate(InPlay = factor(InPlay, levels = c(0, 1)))

deploy <- read_csv("deploy.csv") %>% 
  mutate(SpinRate = as.numeric(SpinRate)) %>% 
  mutate(SpinRate = as.numeric(SpinRate))

## Exploratory Data Analysis
miss_var_summary(training)
# 6 NA values for SpinRate -- must impute

training %>% 
  ggplot(aes(x = Velo, y = InPlay)) +
  geom_boxplot()

training %>% 
  ggplot(aes(x = SpinRate, y = InPlay)) +
  geom_boxplot()

training %>% 
  ggplot(aes(x = HorzBreak, y = InPlay)) +
  geom_boxplot()

training %>% 
  ggplot(aes(x = InducedVertBreak, y = InPlay)) +
  geom_boxplot()

## V-Fold Cross Validation
training_folds <- vfold_cv(training, v = 5, repeats = 3)

## Feature Engineering
training_recipe <- recipe(
  InPlay ~ Velo + SpinRate + HorzBreak + InducedVertBreak, data = training) %>% 
  step_impute_linear(SpinRate) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

training_recipe %>% 
  prep() %>% 
  bake(new_data = NULL)

## Random Forest Model
rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune()
) %>% 
  set_mode("classification") %>% 
  set_engine("ranger", importance = "impurity")

rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 3)))

rf_grid <- grid_regular(rf_params, levels = 3)

rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(training_recipe)

rf_tune <- rf_workflow %>% 
  tune_grid(
    resamples = training_folds,
    grid = rf_grid,
    control = control_stack_resamples()
  )

tune_results <- tibble(
  model_type = c("random forest"),
  tune_info = list(rf_tune),
  assessment_info = map(tune_info, collect_metrics),
  best_model = map(tune_info, ~ select_best(.x, metric = "accuracy"))
)

best_models <- tune_results %>% 
  select(assessment_info) %>% 
  unnest(assessment_info) %>% 
  filter(.metric == "accuracy") %>% 
  arrange(desc(mean))

tune_results %>%
  select(model_type, best_model) %>%
  unnest(best_model)

rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "accuracy"))

rf_results <- fit(rf_workflow_tuned, training)

rf_metric <- metric_set(accuracy)

predictions <- predict(rf_results, new_data = deploy, type = "prob")

deploy_predictions <- deploy %>% 
  mutate(Predicted_InPlay = predictions$.pred_1) %>% 
  select(Predicted_InPlay, everything())

write_csv(deploy_predictions, file = "deploy_predictions.csv")

rf_results %>%
  pull_workflow_fit() %>%
  vip()


