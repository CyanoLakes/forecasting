# Example API query script 
# Downloads all statistics for all dams in subscription, saves to csv
# Author: CyanoLakes (Pty) Ltd

# Login credentials
username <- "Forecasting"
password <- "forecast"

# Specify working directory
wdir <- "/home/mark/Documents/Forecasting/"

# Specify name of output file
file.stats <- "Summary_stats.csv"

# Install and require needed packages (first run only)
#install.packages("httr")
#install.packages("jsonlite")
require("httr")
require("jsonlite")

# Function to query database
query <- function(call, username, password)
	{
	result <- GET(call, authenticate(username, password, type = "basic"))
	result <- content(result, "text")
	result <- fromJSON(result, flatten = TRUE)
	result <- as.data.frame(result, stringsAsFactors=FALSE)
	return(result)
	}

# Function to remove duplicate rows
remove_duplicates <- function(df)
	{
	if (nrow(df)>1) {
		print("Warning! More than one entry for this date.")
		df <- df[1,]  # return first entry
		df$row.names <- NULL
		return(df)}
	else {return(df)}
	}
	# if (df["szenith"][1,1]<df["szenith"][2,1]) {
	# 	df <- df[1,]
	# 	df$row.names <- NULL
	# 	return(df)}
	# else {
	# 	df <- df[2,]
	# 	df$row.names <- NULL
	# 	return(df)}
	# }
	# else {
	# 	return(df)}
	# }

# Open libraries
library("jsonlite")
library("httr")

# API query options
base <- "https://online.cyanolakes.com/api/"
format <- "json"

# Get dams
damscall <- paste(base,"dams","?","format","=", format, sep="")
dams <-query(damscall, username, password)

# Get names
damnames <- dams$name
i <- 1

# First call to get data structure
call1 <- paste(base,"dates/",dams$id[1],"?","format","=", format, sep="")
dates1 <- query(call1, username, password)
call2 <- paste(base,"statistics/",dams$id[1],"/",dates1[1,],"?","format","=", format, sep="")
df <- query(call2, username, password)
df <- remove_duplicates(df)
df$name <- NA

#Loop through dams, date populating dataframe
j <- 1
for(dam.n in dams$id) {
	print(paste("Downloading statistics for ", dam.n))
	datescall <- paste(base,"dates/",dam.n,"?","format","=", format, sep="")
	dates <- query(datescall, username, password)
	
	# using available dates above, find out stats for each date
	for(row in 1:nrow(dates)) {
		statscall <- paste(base,"statistics/",dam.n,"/",dates[row,1],"?","format","=", format, sep="")
		stats <- query(statscall, username, password)
		
		# Remove duplicates
		stats <- remove_duplicates(stats)
		
		# Add to dataframe
		df[j, ] <- stats[1,]
		df[j, "name"] <- damnames[i]
		
		print(paste("Downloaded statistics for ", dates[row,1]))
		j <- j+1
		break
	} 
	i <- i+1
	}

# Unlist coordinates 
df$lat <- NA
df$lon <- NA
for(irow in 1:nrow(df)) {df$lat[irow] <- unlist(df$location.coordinates[irow])[2]}
for(irow in 1:nrow(df)) {df$lon[irow] <- unlist(df$location.coordinates[irow])[1]}
df$location.coordinates <- NULL
df$location.type <- NULL

# Write dataframe to file
write.table(df, file=paste0(wdir, file.stats), sep = ",", row.names=FALSE)
print(paste("Downloaded data. Written to csv file at ", wdir, file.stats))

