X<-sample(c(1:150),replace=TRUE) #sample случайным образом переупорядочивает элементы, переданные в качестве первого аргумента.
xl <- iris[X, 3:5]               #replace=TRUE исключает повторение элементов, если убрать то при каждом 3апуске программы элементы будут на одних и тех же местах.

                                 #xl выборка по 150 случайным числам (от 1 до 150)

colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")         #3адаем цвета 

plot(xl[, 1:2], pch = 21, bg = colors[xl$Species], col= colors[xl$Species], asp = 1) #Рисуем выборку из 150 обьектов

euclideanDistance <- function(u, v)                                                  #Евклидово расстояние 
{ 
  sqrt(sum(u - v)^2) 
}

nn <- function(z, xl)
{
                                       
                                       #Определяем размер выборки
  l <- nrow(xl)                        #Строки 150
  n <- ncol(xl)-1                      #Колонки 2 ( изначально было 3 в выборке xl)
  
  distances <- c()                     #Используем вектор расстояния
  for (i in 1:l)
  {
    distances <- c(distances, euclideanDistance(xl[i, 1:n], z))
  }
 
  xl[order(distances)[1], n+1]
}
                                       #определяем класс и после красим точки
for (ytemp in seq(0, 3, by=0.1)){
  for (xtemp in seq(0, 7, by=0.1)){
    z <- c(xtemp,ytemp)                                          
    class <- nn(z, xl)                                           #Берём класс из функции nn
                                                                 
    points(z[1], z[2], pch = 21, col = colors[class], asp = 1)   #3акрашиваем точки
  }
}