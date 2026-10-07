# Fit the instructor's actual R softImpute model, not a substituted SVD library.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1]); mode <- args[2]
.libPaths(c(file.path(root,"runtime/r-library"),.libPaths()))
suppressPackageStartupMessages(library(softImpute))
data <- read.csv(file.path(root,"data/ratings_for_r.csv"))
pick <- function(name) read.csv(file.path(root,"data",paste0(name,".csv")))$row + 1L
clip <- function(x) pmin(5,pmax(1,x))
predict_fit <- function(fit,eval,user_map,movie_map,mu,bu,bi) {
  u <- match(eval$userID,user_map); m <- match(eval$movieID,movie_map)
  warm <- !is.na(u) & !is.na(m)
  # Unknown IDs cannot be represented by a fit-period latent factor. Use
  # smoothed marginal biases with the training mean for cold cases.
  pu <- bu[match(eval$userID,names(bu))]; pm <- bi[match(eval$movieID,names(bi))]
  pu[is.na(pu)] <- 0;pm[is.na(pm)] <- 0
  pred <- mu + pu + pm
  pred[warm] <- impute(fit,u[warm],m[warm],unscale=TRUE)
  list(pred=clip(pred),warm=warm)
}
run <- function(fit_rows,eval_rows,rank,lambda,prefix) {
  train <- data[fit_rows,]; eval <- data[eval_rows,]
  user_map <- sort(unique(train$userID)); movie_map <- sort(unique(train$movieID))
  if(mode %in% c("teacher","fresh_final")) {
    # Match the notebook's global dense MovieID indexing exactly. An entirely
    # unseen movie column gets the notebook's zero-factor prediction, clipped
    # to one. The improved chronological experiment has an explicit fallback.
    user_map <- seq_len(max(data$userID)); movie_map <- seq_len(max(data$movieID))
  }
  u <- match(train$userID,user_map); m <- match(train$movieID,movie_map)
  mat <- Incomplete(u,m,train$rating)
  # The instructor notebook seeds NumPy, not R. Seed R here for a reproducible
  # new refit. Do not claim the original random initialization is identical.
  set.seed(144)
  start <- proc.time()[3];warnings <- character()
  fit <- withCallingHandlers(softImpute(mat,rank.max=rank,lambda=lambda,maxit=1000),warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
  seconds <- proc.time()[3]-start
  mu <- mean(train$rating)
  bu <- tapply(train$rating-mu,train$userID,sum)/(table(train$userID)+20)
  bi <- tapply(train$rating-mu,train$movieID,sum)/(table(train$movieID)+20)
  predictions <- predict_fit(fit,eval,user_map,movie_map,mu,bu,bi)
  if(mode %in% c("teacher","fresh_final")) {
    predictions$pred <- clip(impute(fit,eval$userID,eval$movieID,unscale=TRUE))
    predictions$warm <- eval$userID %in% train$userID & eval$movieID %in% train$movieID
  }
  write.csv(data.frame(row=eval$row,prediction=predictions$pred,warm=predictions$warm),file.path(root,"results",paste0(prefix,"_predictions.csv")),row.names=FALSE)
  saveRDS(list(fit=fit,user_ids=user_map,movie_ids=movie_map,mu=mu,bu=bu,bi=bi,rank=rank,lambda=lambda),file.path(root,"models",paste0(prefix,".rds")))
  genre_v <- fit$v
  write.csv(data.frame(movieID=movie_map,genre_v),file.path(root,"results",paste0(prefix,"_movie_factors.csv")),row.names=FALSE)
  err <- predictions$pred-eval$rating
  metrics <- data.frame(mode=mode,rank=rank,lambda=lambda,rows=nrow(eval),warm=sum(predictions$warm),RMSE=sqrt(mean(err^2)),MAE=mean(abs(err)),OSR2=1-sum(err^2)/sum((eval$rating-mu)^2),seconds=seconds,warnings=paste(warnings,collapse="; "),package_version=as.character(packageVersion("softImpute")),R_version=R.version.string)
  write.csv(metrics,file.path(root,"results",paste0(prefix,"_fit.csv")),row.names=FALSE)
  print(metrics);flush.console()
  metrics
}
if(mode=="teacher") run(pick("teacher_train"),pick("teacher_test"),8,0,"teacher_rank8")
if(mode=="fresh_final") {
  table <- read.csv(file.path(root,"results/fresh_cv_metrics.csv"))
  best <- table$rank[which.min(table$RMSE)]
  run(pick("teacher_train"),pick("teacher_test"),best,0,"fresh_selected")
}
if(mode=="temporal_grid") {
  result <- list(); j <- 1L
  for(rank in c(4L,8L,12L)) for(lambda in c(0,20)) {
    prefix <- paste0("temporal_r",rank,"_l",lambda)
    result[[j]] <- run(pick("temporal_train"),pick("temporal_validation"),rank,lambda,prefix);j <- j+1L
  }
  table <- do.call(rbind,result)
  write.csv(table,file.path(root,"results/temporal_validation_grid.csv"),row.names=FALSE)
}
if(mode=="temporal_final") {
  table <- read.csv(file.path(root,"results/temporal_validation_grid.csv"))
  best <- table[which.min(table$RMSE),]
  run(c(pick("temporal_train"),pick("temporal_validation")),pick("temporal_test"),best$rank,best$lambda,"temporal_final")
}
if(mode=="random_test_predict") {
  # Save clipped fit-period predictions solely for diagnostic in-sample fit.
  # Hybrid training uses cached OUT-OF-FOLD predictions instead.
  model <- readRDS(file.path(root,"models/teacher_rank8.rds"))
  train <- data[pick("teacher_train"),]
  pred <- predict_fit(model$fit,train,model$user_ids,model$movie_ids,model$mu,model$bu,model$bi)
  write.csv(data.frame(row=train$row,prediction=pred$pred),file.path(root,"results/teacher_insample_predictions.csv"),row.names=FALSE)
}
if(mode=="fresh_cv") {
  ix <- pick("teacher_train"); train <- data[ix,]
  fold <- read.csv(file.path(root,"data/teacher_train.csv"))$fold
  predictions <- matrix(NA_real_,nrow(train),20L)
  fold_log <- list(); j <- 1L
  for(f in 1:20) {
    fit_rows <- which(fold!=f); eval_rows <- which(fold==f)
    mat <- Incomplete(train$userID[fit_rows],train$movieID[fit_rows],train$rating[fit_rows])
    for(rank in 1:20) {
      set.seed(144+100*f+rank); warnings <- character();start <- proc.time()[3]
      fit <- withCallingHandlers(softImpute(mat,rank.max=rank,lambda=0,maxit=1000),warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")})
      pred <- clip(impute(fit,train$userID[eval_rows],train$movieID[eval_rows]))
      predictions[eval_rows,rank] <- pred
      fold_log[[j]] <- data.frame(fold=f,rank=rank,seconds=proc.time()[3]-start,warnings=paste(warnings,collapse="; "));j <- j+1L
    }
    write.csv(do.call(rbind,fold_log),file.path(root,"results/fresh_cv_fit_log.csv"),row.names=FALSE)
    cat("Completed fresh CV fold",f,"of 20\n");flush.console()
  }
  fold_mean <- vapply(1:20,function(f) mean(train$rating[fold!=f]),numeric(1))
  baseline <- fold_mean[fold]
  denom <- sum((train$rating-baseline)^2)
  err <- sweep(predictions,1,train$rating,"-")
  sse <- colSums(err^2)
  table <- data.frame(rank=1:20,RMSE=sqrt(sse/nrow(train)),MAE=colMeans(abs(err)),OSR2=1-sse/denom)
  best <- which.min(table$RMSE)
  write.csv(table,file.path(root,"results/fresh_cv_metrics.csv"),row.names=FALSE)
  write.csv(data.frame(row=train$row,prediction=predictions[,best]),file.path(root,"results/fresh_cv_selected_oof.csv"),row.names=FALSE)
  saveRDS(predictions,file.path(root,"data/fresh_cv_predictions.rds"))
  cat("Fresh 400-fit CV selected rank",best,"\n");print(table)
}
if(mode=="temporal_crossfit") {
  table <- read.csv(file.path(root,"results/temporal_validation_grid.csv"));best <- table[which.min(table$RMSE),]
  order <- order(data$timestamp);n <- nrow(data)
  cut50 <- data$timestamp[order[floor(.50*n)+1L]]
  cut65 <- data$timestamp[order[floor(.65*n)+1L]]
  cut80 <- min(data$timestamp[pick("temporal_validation")])
  run(which(data$timestamp<cut50),which(data$timestamp>=cut50 & data$timestamp<cut65),best$rank,best$lambda,"forward_fold50")
  run(which(data$timestamp<cut65),which(data$timestamp>=cut65 & data$timestamp<cut80),best$rank,best$lambda,"forward_fold65")
}
