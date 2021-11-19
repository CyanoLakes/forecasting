# Example API query script 
# Displays information on all dams in subscription
# Author: CyanoLakes (Pty) Ltd

# Install and require needed packages
#install.packages("httr")
#install.packages("jsonlite")
#require("httr")
#require("jsonlite")

# Open libraries
library("jsonlite")
library("httr")

# Login credentials
username <- "Forecasting"
password <- "forecast"

# API query options
base <- "https://online.cyanolakes.com/api/"
endpoint <- "dams"
format <- "json"

# Specify query
call1 <- paste(base,endpoint,"?","format","=", format, sep="")

# Query API
dams <- GET(call1, authenticate(username,password, type = "basic"))

# Get content and format in JSON
dams <- content(dams, "text")
dams <- fromJSON(dams, flatten = TRUE)

# Convert to dataframe
dams <- as.data.frame(dams)

# View dataframe
View(dams)
