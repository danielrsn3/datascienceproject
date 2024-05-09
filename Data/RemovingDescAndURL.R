# Original dataset was too large to load into VS code, 
# so we decided to remove URL columns and description column 
# using R before using the dataset for the project
# To enhance transparency the R file has been put in the repos
data <- read.csv("/Users/Daniel/Desktop/vehicles.csv")

data$description <- NULL
data$region_url <- NULL
data$url <- NULL
data$image_url <- NULL

write.csv(data, "/Users/Daniel/Desktop/vehicles_Raw.csv", row.names = FALSE)