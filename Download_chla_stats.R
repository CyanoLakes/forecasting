# Example API query script 
# Downloads all chl-a data for all dams in subscription, saves to csv 
# Author: CyanoLakes (Pty) Ltd

# API query options
base <- "https://online.cyanolakes.com/api/"
format <- "json"

# Login credentials
username <- "Forecasting"
password <- "forecast"

# Specify working directory
wdir <- "/home/mark/PycharmProjects/forecasting/"

# Specify output file
file.stats <- "CyanoLakes_chl_stats.csv"

# Install and require needed packages (first run only)
#install.packages("httr")
#install.packages("jsonlite")
#require("httr")
#require("jsonlite")

# Function to query database
query <- function(call, username, password)
{
result <- GET(call, authenticate(username,password, type = "basic"))
result <- content(result, "text", encoding="UTF-8")
result <- fromJSON(result, flatten = TRUE)
return(result)
}

# Open libraries
library("jsonlite")
library("httr")

# Pre-assign data frame
Data <- data.frame(ID=numeric(0), 
name=character(0), 
date=character(0), 
time=character(0),
chla_med=numeric(0),
chla_mean=numeric(0),
chla_max=numeric(0),
chla_min=numeric(0),
chla_std=numeric(0),
chla_cyano=numeric(0),
chla_noncyan=numeric(0),
stringsAsFactors=F)

# Query dams
damscall <- paste(base,"dams","?","format","=", format, sep="")
dams <- query(damscall, username, password )

print(dams)

# Convert to dataframe
dams <- as.data.frame(dams)

# Get names
damnames <- dams$name
i <- 1

#Loop through dams getting statistics for each
for(dam.n in dams$id) {
	print(paste("Downloading statistics for ", dam.n))

	# Get dates
	datescall <- paste(base,"dates/",dam.n,"?","format","=", format, sep="")
	dates <- query(datescall, username, password)
	
	# using available dates above, find out stats for each date
	for(date.n in dates) {
		# Get stats for date
		statscall <- paste(base,"statistics/",dam.n,"/",date.n,"?","format","=", format, sep="")
		stats <- query(statscall, username, password)
		stats <- as.data.frame(stats)
		
		# If more than one row, choose first row
		if (nrow(stats)>1) {
			print("Warning! More than one entry for this date.")
			stats <- stats[1,]  # return first entry
		}
		
		## Populate data frame (note order must be preserved)
		Data[nrow(Data)+1, ] <- c(
		dam.n,
		damnames[i],
		stats$date,
		stats$time,
		stats$chla_med,
		stats$chla_mean,
		stats$chla_max,
		stats$chla_min,
		stats$chla_std,
		stats$chla_cyano,
		stats$chla_noncyano
		)

		print(paste("Downloaded statistics for ", date.n))
	} 
	i <- i+1
	}

# Write dataframe to file
write.table(Data, file=paste0(wdir, file.stats), sep = ",", row.names=FALSE)
print(paste("Downloaded data. Written to csv file at ", wdir, file.stats))
