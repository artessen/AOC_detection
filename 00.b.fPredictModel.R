# Loading the required libraries ===============================================
library(data.table)
library(caret)
library(xgboost)
library(Metrics)
library(ggplot2)
library(ggplotify)
library(parallel)
library(gridExtra)
library(lubridate)
library(reshape2)
library(haven)
library(dplyr)
library(stringr)
library(matrixStats)
library(tidyverse)
# ==============================================================================

# Reading the input data =======================================================
# Reading the database
dfData = fread("dfDatabaseProd_sample.csv")

# Defining the ID and Target columns
sIDs = c("X","Y","Time","Month","Year","Day")
sTgts = c("ColdS", "Drought", "HeatW", "HotDry", "RainD", "RainS", "TSumS", "TSumD")

# Defining the possible Feature columns
sERA5mean = c("msl", "sp", "t2m", "tmax", "tmin", "tp", "z500")
sERA5anom = c("msl", "sp", "t2m", "tmax", "tmin", "tp", "z500")
sERA5spec = c("daystmax25a30C", "daystmax30C", 
              "daystmin0C", "daystminm10C", "daystminm18C", 
              "daystp01mm", "daystp30mm")
sSPI = c("spi01","spi03","spi06","spi09","spi12")
sCEMS = c("smant", "fpanv")
sERA5mean = paste(sERA5mean, "_mean", sep="")
sERA5anom = paste(sERA5anom, "_anom", sep="")
# ==============================================================================

# For every target variable ====================================================
# Training the model for the selected target variable
for (i in 1:length(sTgts)) {
  
  # Progress message
  sTgt = sTgts[i]
  message("Processing target variable: ", sTgt)
  
  # Getting the features
  sFeats = readRDS(list.files(paste("./",sTgt,sep=""), ".rds", full.names=T)[1])$feature_names
  
  # Creating the database to extract the data and results
  iCols = which(names(dfData) %in% c(sIDs, sFeats, sTgts))
  dfDataset = dfData[,..iCols] 
  dfRes = dfDataset[,..sIDs]
  dfDataset = na.omit(dfDataset)
  iCols = which(names(dfDataset) %in% c(sIDs))
  dfIDs = dfDataset[,..iCols] 
  
  # Preparing the data for a xgb model
  iCols = which(!(names(dfDataset) %in% sIDs))
  dfXGB = xgb.DMatrix(data=as.matrix(dfDataset[,..iCols]))
  
  # Computing the mean and st_dev values of the predictions
  sModels = list.files(paste("./",sTgt,sep=""), ".rds", full.names=T)
  sModels = sModels[1:1000]
  lModels = split(sModels, ceiling(seq_along(sModels)/100))
  fPredict = function(sModel) { predict(readRDS(sModel), newdata=dfXGB) }
  dfDatasetRes = NULL
  pbi = 0; pb = txtProgressBar(min=0, max=length(lModels), initial=0, style=3)
  for (sModels in lModels) {
    mDatasetRes = lapply(sModels, fPredict)
    mDatasetRes = matrix(unlist(mDatasetRes), nrow=NROW(dfIDs), ncol=length(sModels), byrow=F)
    dfDatasetRes = cbind(dfDatasetRes, mDatasetRes)
    rm(mDatasetRes); gc()
    pbi = pbi+1; setTxtProgressBar(pb,pbi)
  }
  dfDatasetRes = data.frame(pred_mean=rowMeans(dfDatasetRes), 
                            pred_sd=rowSds(dfDatasetRes), 
                            count_50=rowMeans(dfDatasetRes>0.50), 
                            count_90=rowMeans(dfDatasetRes>0.90),
                            count_95=rowMeans(dfDatasetRes>0.95),
                            count_99=rowMeans(dfDatasetRes>0.99)); gc()
  
  # Consolidating the results
  dfDatasetRes = cbind(dfDataset[,..sIDs], dfDatasetRes)
  names(dfDatasetRes)[NCOL(dfDatasetRes)-5] = paste("pred_",sTgt,sep="")
  names(dfDatasetRes)[NCOL(dfDatasetRes)-4] = paste("sd_",sTgt,sep="")
  names(dfDatasetRes)[NCOL(dfDatasetRes)-3] = paste("cnt50_",sTgt,sep="")
  names(dfDatasetRes)[NCOL(dfDatasetRes)-2] = paste("cnt90_",sTgt,sep="")
  names(dfDatasetRes)[NCOL(dfDatasetRes)-1] = paste("cnt95_",sTgt,sep="")
  names(dfDatasetRes)[NCOL(dfDatasetRes)] = paste("cnt99_",sTgt,sep="")
  
  # Consolidating the results
  dfRes = merge(dfRes, dfDatasetRes, by=sIDs, all.x=T)
  
  # Storing the predictions of the model to the disk
  fwrite(dfRes, paste("./xgb_model_predictions_",sTgt,".csv",sep=""))
  
  # Cleaning-up
  rm(dfDataset, dfRes, dfIDs, dfXGB, sModels, dfDatasetRes, dfTgt)
  gc()
}
# ==============================================================================
