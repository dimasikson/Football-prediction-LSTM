
leagueCode = c(
  'E0'
  # 'D1'
  # 'I1'
  # 'SP1'
  # 'F1'
  # 'N1'
  # 'P1'
  # 'T1'
)[1]

loadData = function(leagueCode, fileNum){
  
  fnames = c()
  
  for (fname in list.files('Football prediction Python/predictedOutput/')){
    if (fname %>% startsWith(leagueCode) == TRUE){
      fnames = c(paste0('Football prediction Python/predictedOutput/',fname), fnames)
    }
  }
  
  df = fnames[fileNum] %>% read_csv()
  
  df = df[df$T_GamesPlayed_H+df$T_GamesPlayed_A >= 6,]
  
  return(df)

}

generateReturn = function(df, drawOddsCutOff, predictType){
  
  if (predictType=='result'){
    df$PredictedResult = ifelse(
      test = pmax(df$HPredict,df$APredict) < drawOddsCutOff,
      yes = 1/3,
      no = ifelse(df$HPredict>=df$APredict, 1, 0)
    )
  } else if (predictType=='goals'){
    df$PredictedResult = ifelse(
      test = abs(10*df$HPredict_Goals-10*df$APredict_Goals) < drawOddsCutOff,
      yes = 1/3,
      no = ifelse(df$HPredict_Goals>=df$APredict_Goals, 1, 0)
    )
  } else if (predictType=='goaldif'){
    df$PredictedResult = ifelse(
      test = abs(10*df$Predict_Goaldif) < drawOddsCutOff,
      yes = 1/3,
      no = ifelse(df$Predict_Goaldif>0, 1, 0)
    )
  }

  df$correctPredict = ifelse(df$Points_H == df$PredictedResult,1,0)
  
  df$Return = ifelse(df$Points_H==1,df$HomeOdds,
                     ifelse(df$Points_H==1/3,df$DrawOdds,df$AwayOdds))*20*df$correctPredict-1
  
  return(df)

}

predictedDf = loadData(leagueCode, 1)
predictedDf = generateReturn(predictedDf, 0.4, 'result')

summaryStats = sqldf(
'
  select
    PredictedResult,
    sum(1) as Games,
    sum(correctPredict) as correctPredict,
    sum(Return) as Return,
    100*round(sum(correctPredict)/sum(1),3) as Accuracy,
    100*round(sum(Return)/sum(1),3) as Return_pct
  from predictedDf
  group by 1
  order by 1 desc
')

summaryStats

plotDf = matrix(ncol = 5, nrow = 0) %>% as.data.frame(); colnames(plotDf) = c('i', 'H', 'D', 'A', 'TTL')

for (i in seq(0.1,0.9,0.01)){
  predictedDf = generateReturn(predictedDf, i, 'result')
  
  summ = sqldf(
  '
    select
      round(PredictedResult,2) as PredictedResult,
      100*round(sum(Return)/sum(1),3) as Return_pct
    from predictedDf
    group by 1
    union
    select
      100 as PredictedResult,
      100*round(sum(Return)/sum(1),3) as Return_pct
    from predictedDf
    group by 1
  ')
  
  tempDf = c(i,0,0,0,0) %>% t() %>% as.data.frame()
  
  colValues = c(1,0.33,0,100)
  
  for (j in seq(2,5)){
    tempDf[1,j] = coalesce(c(summ$Return_pct[summ$PredictedResult==colValues[j-1]],0),0)[1]
  }
  
  colnames(tempDf) = c('i', 'H', 'D', 'A', 'TTL')
  plotDf = rbind(plotDf,tempDf)

}

ggplot(plotDf, aes(x=i))+
  geom_line(aes(y=H), color='green')+
  geom_line(aes(y=D), color='blue')+
  geom_line(aes(y=A), color='red')+
  geom_line(aes(y=TTL), color='black', linetype = 'longdash')+
  ylim(-50,50)

