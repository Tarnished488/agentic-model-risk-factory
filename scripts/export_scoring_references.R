# Appended to the vetted original training cells by prepare_scoring_references.py.
# Portable model parameters and preprocessing only; no executable RDS is consumed by Python.
library(jsonlite)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_json_config <- function(x,path) write_json(x,path,auto_unbox=TRUE,digits=NA,pretty=TRUE,na='null')
precise <- function(x) vapply(x,function(v) sprintf('%.17g',v),character(1))
if (group == 'gmsc_fastwalker') {
  f <- fit.gbm$finalModel
  full <- readr::read_csv(file.path(data_dir,'cs-training.csv'),show_col_types=FALSE) %>% dplyr::rename(Id=1) %>% data_prep()
  holdout_raw <- full %>% dplyr::filter(!id %in% tr$id)
  holdout_x <- recipes::bake(rec,holdout_raw)
  prob <- predict(fit.gbm,holdout_x,type='prob')$bad
  trees <- lapply(seq_len(fit.gbm$bestTune$n.trees), function(i) {
    t<-gbm::pretty.gbm.tree(f,i)
    list(split_var=t$SplitVar,code=precise(t$SplitCodePred),left=t$LeftNode,right=t$RightNode,
         missing=t$MissingNode,prediction=precise(t$Prediction))
  })
  write_json_config(list(initF=precise(f$initF),trees=trees),file.path(out,'model.json'))
  config<-list(kind='gbm',data_file='cs-training.csv',target='SeriousDlqin2yrs',
    feature_order=f$var.names,impute=as.list(precise(rec$steps[[1]]$means)),center=as.list(precise(rec$steps[[2]]$means)),scale=as.list(precise(rec$steps[[3]]$sds)),
    limitations=c('Original probability column 2 was good: this adapter explicitly returns bad.',
                  'Original CV index and pre-CV preprocessing retained; review resampling design.'))
  write.csv(data.frame(row_id=as.integer(tr$id)),file.path(out,'train_ids.csv'),row.names=FALSE)
  write.csv(data.frame(row_id=as.integer(holdout_raw$id),probability=prob,label=as.integer(holdout_raw$target=='bad')),file.path(out,'holdout_predictions.csv'),row.names=FALSE)
} else if (group == 'gmsc_lightgbm') {
  lightgbm::lgb.save(bst,file.path(out,'model.txt'))
  config<-list(kind='lightgbm',data_file='cs-training.csv',target='SeriousDlqin2yrs',feature_order=colnames(test$data),
    limitations=c('Original notebook recomputes means within the scoring batch, so scores can depend on batch composition.',
                  'Age-zero correction is applied to training only, as in the original.'))
  write.csv(data.frame(row_id=full$X[train_idx]),file.path(out,'train_ids.csv'),row.names=FALSE)
  write.csv(data.frame(row_id=TestData$X,probability=pred,label=as.integer(TestData2$SeriousDlqin2yrs)),file.path(out,'holdout_predictions.csv'),row.names=FALSE)
} else if (group == 'german_xgboost') {
  r<-workflows::extract_recipe(final_xgb_fit,estimated=TRUE)
  x<-recipes::bake(r,new_data=test) %>% dplyr::select(-default)
  e<-workflows::extract_fit_engine(final_xgb_fit)
  xgboost::xgb.save(e,file.path(out,'model.json'))
  native<-predict(e,as.matrix(x))
  d0<-max(abs(native-pred_xgb$.pred_bad));d1<-max(abs(1-native-pred_xgb$.pred_bad))
  stopifnot(min(d0,d1)<1e-5)
  categories<-lapply(names(r$steps[[3]]$levels),function(name){
    values<-attr(r$steps[[3]]$levels[[name]],'values')
    list(levels=values,columns=recipes:::dummy_names(name,values[-1],ordinal=FALSE))
  });names(categories)<-names(r$steps[[3]]$levels)
  config<-list(kind='xgboost',data_file='german_credit_data.csv',target='Risk',target_mapping=list(good=0L,bad=1L),
    feature_order=colnames(x),categories=categories,center=as.list(r$steps[[4]]$means),scale=as.list(r$steps[[4]]$sds),
    high_loan_cutoff=unname(quantile(credit_raw$credit_amount,.75)),complement_bad_probability=d1<d0,
    limitations=c('High-loan quantile is calculated before splitting in the original; review preprocessing look-ahead.',
                  'SMOTE is preserved in R training, not applied to held-out Python scoring.',
                  'Original reported AUC used the wrong event direction; bad is explicitly mapped to 1.'))
  write.csv(data.frame(row_id=split$in_id-1L),file.path(out,'train_ids.csv'),row.names=FALSE)
  write.csv(data.frame(row_id=setdiff(seq_len(nrow(credit)),split$in_id)-1L,probability=pred_xgb$.pred_bad,label=as.integer(pred_xgb$truth=='bad')),file.path(out,'holdout_predictions.csv'),row.names=FALSE)
} else stop('Unknown model group')
write_json_config(config,file.path(out,'preprocessing.json'))
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
cat('Portable reference exported:',group,'\n')
