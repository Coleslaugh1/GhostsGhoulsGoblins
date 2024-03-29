
# Setting up work environment and libraries -------------------------------

#setwd(dir = "C:/Users/camer/Documents/Stat 348/GhostsGhoulsGoblins/")

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(doParallel)
library(discrim)
library(themis)
library(stacks)
library(kernlab)
library(keras)
library(lightgbm)
library(bonsai)
library(dbarts)

# Parallel Processing

# library(doParallel)
# parallel::detectCores() #How many cores do I have?
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)
# #code
# stopCluster(cl)

# Initial reading of the data

rawdata <- vroom(file = "train.csv") %>%
  mutate(type=factor(type))
test_input <- vroom(file = "test.csv")

# missing <- vroom(file = "trainWithMissingValues.csv") %>%
#   mutate(type=factor(type))

# Recipes

my_recipe <- recipe(type ~ ., data = rawdata) %>%
  update_role(id, new_role="id") %>% 
  step_mutate_at(color,fn = factor) %>% 
  step_dummy(color) %>% 
  step_rm(id) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) %>% 
  step_normalize(all_numeric_predictors())
  
my_recipe <- recipe(type ~ ., data = rawdata) %>%
  update_role(id, new_role="id") %>% 
  step_mutate_at(color,fn = factor) %>% 
  step_rm(color) %>% 
  step_rm(id) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) #%>% 
  #step_normalize(all_numeric_predictors())


prep_recipe <- prep(my_recipe)
baked_data <- bake(prep_recipe, new_data = rawdata)

# my_recipe <- recipe(type ~ ., data = missing) %>%
#   step_impute_mean(all_numeric_predictors())
# 
# prep_recipe <- prep(my_recipe)
# baked_data <- bake(prep_recipe, new_data = missing)
# 
# rmse_vec(rawdata[is.na(missing)], baked_data[is.na(missing)])

# Write and read function

format_and_write <- function(predictions, file){
  final_preds <- predictions %>%
    mutate(type = .pred_class) %>%
    mutate(id = test_input$id) %>%
    dplyr::select(id, type)

  vroom_write(final_preds,file,delim = ",")
  #save(file="./MyFile.RData", list=c("object1", "object2",...))
}


# svm ---------------------------------------------------------------------

# svm_model <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svm_model)
# 
# tuning_grid <- grid_regular(rbf_sigma(),
#                             cost(),
#                             levels = 4)
# 
# folds <- vfold_cv(rawdata, v = 10, repeats=1)
# 
# cl <- makePSOCKcluster(4)
# registerDoParallel(cl)
# CV_results <- svm_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# final_svm_wf <-
#   svm_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# 
# svm_predictions <- final_svm_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(svm_predictions, "svm_preds.csv")


# Neural Nets -------------------------------------------------------------

# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50) %>% #or 100 or 250
#   set_engine("nnet") %>% #verbose = 0 prints off less
#   set_mode("classification")
# 
# nn_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(nn_model)
# 
# tuning_grid <- grid_regular(hidden_units(range=c(1, 10)),
#                             levels=4)
# 
# folds <- vfold_cv(rawdata, v = 10, repeats=1)
# 
# # cl <- makePSOCKcluster(10)
# # registerDoParallel(cl)
# CV_results <- nn_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# # stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# final_nn_wf <-
#   nn_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# 
# nn_predictions <- final_nn_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(nn_predictions, "nn_preds.csv")


# CV_results %>% collect_metrics() %>%
# filter(.metric=="accuracy") %>%
# ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## CV tune, finalize and predict here and save results
## This takes a few min (10 on my laptop) so run it on becker if you want


# Boosting & BART ---------------------------------------------------------

# Boost
# boost_model <- boost_tree(tree_depth=tune(),
#                           trees=tune(),
#                           learn_rate=tune()) %>%
#   set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
#   set_mode("classification")
# 
# boost_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(boost_model)
# 
# tuning_grid <- grid_regular(tree_depth(),
#                             trees(),
#                             learn_rate(),
#                             levels=4)
# 
# folds <- vfold_cv(rawdata, v = 10, repeats=1)
# 
# cl <- makePSOCKcluster(10)
# registerDoParallel(cl)
# CV_results <- boost_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# final_boost_wf <-
#   boost_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# 
# boost_predictions <- final_boost_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(boost_predictions, "boost_preds.csv")

# Bart
# bart_model <- parsnip::bart(trees=tune()) %>% # BART figures out depth and learn_rate
#   set_engine("dbarts") %>% # might need to install
#   set_mode("classification")
# 
# bart_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(bart_model)
# 
# tuning_grid <- grid_regular(trees(),
#                             levels=4)
# 
# folds <- vfold_cv(rawdata, v = 10, repeats=1)
# 
# cl <- makePSOCKcluster(10)
# registerDoParallel(cl)
# CV_results <- bart_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# final_bart_wf <-
#   bart_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# 
# bart_predictions <- final_bart_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(bart_predictions, "bart_preds.csv")

# Discriminant Analysis linear ---------------------------------------------------

# library(mda)
# dal_model <- discrim_linear(penalty = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("mda")
# 
# dal_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(dal_model)
# 
# tuning_grid <- grid_regular(penalty(),
#                             levels = 4)
# 
# folds <- vfold_cv(rawdata, v = 10, repeats=1)
# 
# # cl <- makePSOCKcluster(4)
# # registerDoParallel(cl)
# CV_results <- dal_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# # stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# # final_dal_wf <-
# #   dal_workflow %>%
# #   fit(data=rawdata)
# 
# final_dal_wf <-
#   dal_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# dal_predictions <- final_dal_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(dal_predictions, "dal_preds.csv")

# Discriminant Analysis flexible ---------------------------------------------------

# library(earth)
# daf_model <- discrim_flexible(num_terms = tune(),
#                             prod_degree = tune(),
#                             prune_method = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("earth")
# 
# daf_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(daf_model)
# 
# tuning_grid <- grid_regular(num_terms(range=c(1, 10)),
#                             prod_degree(),
#                             prune_method(),
#                             levels = 4)
# 
# folds <- vfold_cv(rawdata, v = 10, repeats=1)
# 
# # cl <- makePSOCKcluster(4)
# # registerDoParallel(cl)
# CV_results <- daf_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# # stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# # final_dal_wf <-
# #   dal_workflow %>%
# #   fit(data=rawdata)
# 
# final_daf_wf <-
#   daf_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# daf_predictions <- final_daf_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(daf_predictions, "daf_preds.csv")

# Discriminant Analysis regularized ---------------------------------------------------

# library(klaR)
# dar_model <- discrim_regularized(frac_common_cov = tune(),
#                                  frac_identity = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("klaR")
# 
# dar_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(dar_model)
# 
# tuning_grid <- grid_regular(frac_common_cov(),
#                             frac_identity(),
#                             levels = 4)
# 
# folds <- vfold_cv(rawdata, v = 5, repeats=1)
# 
# # cl <- makePSOCKcluster(4)
# # registerDoParallel(cl)
# CV_results <- dar_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# # stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# # final_dal_wf <-
# #   dal_workflow %>%
# #   fit(data=rawdata)
# 
# final_dar_wf <-
#   dar_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# dar_predictions <- final_dar_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(dar_predictions, "dar_preds.csv")

# Discriminant Analysis Quad ---------------------------------------------------

# library(sparsediscrim)
# daq_model <- discrim_quad(regularization_method = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("sparsediscrim")
# 
# daq_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(daq_model)
# 
# tuning_grid <- grid_regular(regularization_method(),
#                             levels = 4)
# 
# folds <- vfold_cv(rawdata, v = 4, repeats=1)
# 
# # cl <- makePSOCKcluster(4)
# # registerDoParallel(cl)
# CV_results <- daq_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(accuracy))
# # stopCluster(cl)
# 
# bestTune <- CV_results %>%
#   select_best("accuracy")
# 
# # final_dal_wf <-
# #   dal_workflow %>%
# #   fit(data=rawdata)
# 
# final_daq_wf <-
#   daq_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=rawdata)
# 
# daq_predictions <- final_daq_wf %>%
#   predict(new_data = test_input, type="class")
# 
# format_and_write(daq_predictions, "daq_preds.csv")

# Naive Bayes -------------------------------------------------------------

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

folds <- vfold_cv(rawdata, v = 5, repeats=1)

#cl <- makePSOCKcluster(10)
#registerDoParallel(cl)
CV_results <- nb_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy))
#stopCluster(cl)

bestTune <- CV_results %>%
  select_best("accuracy")

final_nb_wf <-
  nb_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=rawdata)


nb_predictions <- final_nb_wf %>%
  predict(new_data = test_input, type="class")

format_and_write(nb_predictions, "nb_preds.csv")
