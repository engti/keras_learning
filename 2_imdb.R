## load lib
  library(keras)
  
## load data
  imdb <- dataset_imdb(num_words = 10000)
  c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb
  
  ## how to get the words back
  word_index <- dataset_imdb_word_index()
  reverse_word_index <- names(word_index)
  names(reverse_word_index) <- word_index
  decoded_review <- sapply(train_data[[100]], function(index) {
    word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
    if (!is.null(word)) word else "?"
  })
  
  ## one hot encode the review list 
  ## this is so we can use it in the neural network that needs tensors
  vectorize_sequences <- function(sequences, dimension = 10000) {
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    for (i in 1:length(sequences))
      results[i, sequences[[i]]] <- 1
    results
  }
  
  x_train <- vectorize_sequences(train_data)
  x_test <- vectorize_sequences(test_data)
  
  ## convert train integers to numeric
  y_train <- as.numeric(train_labels)
  y_test <- as.numeric(test_labels)
  
  ## start the model
  model <- keras_model_sequential() %>%
    layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  ## setting validation unit
  val_indices <- 1:10000
  
  x_val <- x_train[val_indices,]
  partial_x_train <- x_train[-val_indices,]
  
  y_val <- y_train[val_indices]
  partial_y_train <- y_train[-val_indices]

  ## train model
  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
  )
  
  ## plot history object
  plot(history)
  
  ## turn to data frame for other uses
  history_df <- as.data.frame(history)

  
  ## try again to reduce loss
  model <- keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy")
  )
  
  model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
  results <- model %>% evaluate(x_test, y_test)
    
  model %>% predict(x_test[1:20,])
  
  