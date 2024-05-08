data <- read.csv("/Users/Daniel/Desktop/vehicles.csv")

data$description <- NULL
data$region_url <- NULL
data$url <- NULL
data$image_url <- NULL

write.csv(data, "/Users/Daniel/Desktop/vehicles_Raw.csv", row.names = FALSE)