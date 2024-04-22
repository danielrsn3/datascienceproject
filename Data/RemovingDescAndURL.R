data <- read.csv("/Users/Daniel/Desktop/vehicles.csv")

data$description <- NULL
data$region_url <- NULL
data$url <- NULL
data$image_url <- NULL

write.csv(data, "/Users/Daniel/Desktop/vehicles_no_url.csv", row.names = FALSE)
