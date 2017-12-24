## load lib
  library(keras)
  library(dplyr)

## load data
  mnist <- dataset_mnist()
  
  train_images <- mnist$train$x
  train_labels <- mnist$train$y
  test_images <- mnist$test$x
  test_labels <- mnist$test$y  

  str(train_labels)  
  
  ## build the model
  network <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
    layer_dense(units = 10, activation = "softmax")
  
  network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  ## train the model
  train_images <- array_reshape(train_images, c(60000, 28 * 28))
  train_images <- train_images / 255
  
  test_images <- array_reshape(test_images, c(10000, 28 * 28))
  test_images <- test_images / 255
  
  ## train the labels  
  train_labels <- to_categorical(train_labels)
  test_labels <- to_categorical(test_labels)
  
  ## fit model
  network %>% fit(train_images, train_labels, epochs = 10, batch_size = 128)
  
  ## check accuracy
  metrics <- network %>% evaluate(test_images, test_labels)
  
  ## predict
  network %>% predict_classes(test_images[1:30,])

  
    