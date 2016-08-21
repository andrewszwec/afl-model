## AFL model

require(xgboost)

setwd("/Users/andrew/Documents/R/afl-model")
df <- read.csv("data.csv")

df$Date <- as.Date(df$Date)
df$year <- as.numeric(format(df$Date, "%Y")) 

# filter for 2013 onwards
dd <- df[which(df$year >= 2013 ), ]

#weekdays(dd$Date)
dd$id <- as.numeric(row.names(dd))
dd$day.of.week <- as.POSIXlt(dd$Date)$wday
dd$month.of.year <- as.numeric(format(dd$Date, "%m"))
dd$hour.of.day <- substr( as.character(dd$Kick.Off..local), 1, 2)

my.dates <- subset(dd, select=c(day.of.week, month.of.year, hour.of.day))

# One hot encode Home Team
dd0 <- subset(dd,select=c(Home.Team))
Home.Team.One.Hot <- model.matrix(~.-1,dd0)
# One hot encode Away Team
dd0 <- subset(dd,select=c(Away.Team))
Away.Team.One.Hot <- model.matrix(~.-1,dd0)
# One hot encode Venue
dd0 <- subset(dd,select=c(Venue))
Venue.One.Hot <- model.matrix(~.-1,dd0)

# select cols to keep from original matrix
keep.cols <- subset(dd, select = -c(id,Date,Kick.Off..local.,Home.Team, Away.Team, 
                                           Venue,day.of.week, month.of.year, hour.of.day, Play.Off.Game., year,
                                    Total.Score.Open,	Total.Score.Min,	Total.Score.Max,	Total.Score.Close,	
                                    Total.Score.Over.Open,	Total.Score.Over.Min,	Total.Score.Over.Max,	Total.Score.Over.Close,	
                                    Total.Score.Under.Open,	Total.Score.Under.Min,	Total.Score.Under.Max,
                                    Total.Score.Under.Close,
                                    Away.Score, 
                                    Home.Score, 
                                    Home.Goals, 
                                    Away.Goals
                                    ))
keep.cols <- scale(keep.cols)  

# make some other features
logs <- log(keep.cols, )
squared <- keep.cols ^2

# Make model ready matrix
dd1 <- cbind(keep.cols, my.dates, Home.Team.One.Hot, Away.Team.One.Hot, Venue.One.Hot)

## Add Target Variables
dd1$home.team.win <- ifelse(dd1$Home.Score > dd1$Away.Score, 1, 0 )
dd1$away.team.win <- ifelse(dd1$Home.Score < dd1$Away.Score, 1, 0 )

## Make col names no spaces
names(dd1) <- gsub('[ \\s\\+\\-\\\\\\<\\>]','.',names(dd1), ignore.case = TRUE, perl = TRUE) 

## Remove rows with NA
dd1 <- dd1[complete.cases(dd1), ]

# write.csv(dd1, file="dd1.csv")

# Split the data into a model training and test set used to measure the performance of the algorithm
set.seed(975)
inTrain     = createDataPartition(dd1$home.team.win, p = 0.7)[[1]]
training    = dd1[ inTrain,]      # 70% of records
testing     = dd1[-inTrain,]      # 30% of reocrds


##########################################################
## Build parallel model 
##########################################################
library(randomForest)

ntree = 200 
# time <- system.time({
#   xg_fit <- xgboost(data = subset(training, select=-c(home.team.win, away.team.win)), 
#                     label = subset(training, select=c(home.team.win)), max.depth = 2,
#                     eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
# })

data <- subset(training, select=-c(home.team.win, away.team.win))
label <- as.factor( subset(training, select=c(home.team.win))$home.team.win )
rf_fit <- randomForest( x=data, y = label, ntree = ntree,importance=TRUE)


# Use the model to make a prediction about whether passengers in the test set survived
pred <- predict(rf_fit, newdata=testing)

# compare the prediction of survival with the observation
results <- data.frame(observations=testing$home.team.win, predictions=pred)
results$observations <- as.numeric(results$observations)  
results$predictions <- as.numeric(as.character(results$predictions))

# Print a confusion matrix to view the results
#install.packages("e1701")
#require(e1701)
confusionMatrix(results$predictions, results$observations)

# Look at the contribution of variables to the model
a <- data.frame(rf_fit$importance)
a <- a[order(a$X1,decreasing = TRUE),]
aa <- head(a, n=20L)
aa$var.name <- as.factor(row.names(aa))

# Plot on graph
require(ggplot2)
ggplot(data=aa, aes(x=var.name, y=X1, fill=var.name)) +
  geom_bar(colour="black", stat="identity") +
  guides(fill=FALSE)  + xlim(rev(levels(aa$var.name)))



