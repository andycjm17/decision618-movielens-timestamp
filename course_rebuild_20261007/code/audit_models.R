# Independent check of saved predictions against both impute() and U D V'.
args <- commandArgs(trailingOnly=TRUE)
root <- normalizePath(args[1])
.libPaths(c(file.path(root,"runtime/r-library"),.libPaths()))
suppressPackageStartupMessages(library(softImpute))
d <- read.csv(file.path(root,"data/ratings_for_r.csv"))
out <- list()
for (prefix in c("teacher_rank8","fresh_selected","temporal_final")) {
  m <- readRDS(file.path(root,"models",paste0(prefix,".rds")))
  p <- read.csv(file.path(root,"results",paste0(prefix,"_predictions.csv")))
  q <- d[match(p$row,d$row),]
  u <- match(q$userID,m$user_ids); v <- match(q$movieID,m$movie_ids)
  ix <- which(!is.na(u) & !is.na(v))
  package <- impute(m$fit,u[ix],v[ix],unscale=TRUE)
  manual <- rowSums(sweep(m$fit$u[u[ix],,drop=FALSE],2,m$fit$d,"*") * m$fit$v[v[ix],,drop=FALSE])
  clip <- function(x) pmin(5,pmax(1,x))
  stopifnot(max(abs(package-manual)) < 1e-10)
  stopifnot(max(abs(clip(manual)-p$prediction[ix])) < 1e-10)
  cold <- which(is.na(u) | is.na(v))
  cold_error <- 0
  if(length(cold)) {
    pu <- m$bu[match(q$userID[cold],names(m$bu))]; pu[is.na(pu)] <- 0
    pi <- m$bi[match(q$movieID[cold],names(m$bi))]; pi[is.na(pi)] <- 0
    cold_error <- max(abs(clip(m$mu+pu+pi)-p$prediction[cold]))
    stopifnot(cold_error < 1e-10)
  }
  out[[prefix]] <- data.frame(model=prefix,rows=nrow(q),factor_dot_max_error=max(abs(package-manual)),saved_max_error=max(abs(clip(manual)-p$prediction[ix])),cold_rows=length(cold),cold_max_error=cold_error)
}
write.csv(do.call(rbind,out),file.path(root,"results/independent_r_audit.csv"),row.names=FALSE)
print(do.call(rbind,out))
