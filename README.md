# Машинное обучение
***
### Метрические алгоритмы
Метрические алгоритмы классифицируют объекты в зависимости от их сходства, а для оценки сходства объектов используют функцию расстояния, называемую метрикой. Чем меньше расстояние, тем больше объекты похожи друг на друга.
***
##### 1NN
 Этот алгоритм является частным случаем KNN. Использя его мы измеряем расстояние классифицируемого обьекта до всех элементов выборки.
###### Функция для 1NN
```R
nn <- function(z, xl) 
{                       #Определяем размер выборки
  l <- nrow(xl)         #Строки 150       
  n <- ncol(xl)-1       #Колонки 2 ( изначально было 3 в выборке xl)
  distances <- c()      #Используем вектор расстояния
  for (i in 1:l)
  {
    distances <- c(distances, euclideanDistance(xl[i, 1:n], z))
  }
  xl[order(distances)[1], n+1]
} 
```

###### Получаем следующую карту классификации
![Иллюстрация к проекту](https://github.com/MinaevYuriy/SMPR/blob/master/1G_7rOwyOS4.jpg)

***
### Баесовские алгоритмы
Байесовский подход является классическим в теории распознавания образов и лежит в основе многих методов.
```R
naive = function(x, Py, mu, sigm, m, n) { #функция наивного байесовского классификатора
  mina <- matrix(c('setosa','versicolor', 'virginica', 0, 0, 0), nrow = 3, ncol = 2) #3 класса, 2 признака
  scores = rep(0, m) 
  for (i in 1:m) {
    scores[i] = Py[i] # присваиваем восстановленные плотности
    for (j in 1:n){
      N=1/sqrt(2*pi)/sigm[i,j]*exp(-1/2*(x[j]-mu[i,j])^2/sigm[i,j]^2) #гауссовское нормальнное распределение
      # следуем  оптимальному байесовскому решающему правилу
      scores[i] = scores[i] * N 
    }
    mina[i,2]=scores[i]
  }
  class <- mina[,1][which.max(mina[,2])]
}

```
###### Получаем следующую карту классификации
![Иллюстрация к проекту](https://github.com/MinaevYuriy/SMPR/blob/master/yN3XIGM0m0I.jpg)
    
      
1. хорошо разобрался в том, как находить лицо/лица на фото
	для детектирования лица используется класс get_frontal_face_detector() из библиотеки dlib. В его основе лежит алгоритм HOG. https://www.google.com/search?client=safari&rls=en&q=%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC+HOG+%D0%BE%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5&ie=UTF-8&oe=UTF-8 . 
	https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78 здесь описание всего твоего алгоритма, до блока step1 можешь взять введение, сам блок step1 можешь поспользовать для описание первой части, там будет как раз про алгоритм HOG&
2. нашел код, в котором реализован поворот лица, сейчас разбираешься в нем
3. лица будешь переводить в числа с помощью модели nn4.small2.v1 из https://cmusatyalab.github.io/openface/models-and-accuracies/#pre-trained-models , так как она показывает лучшее качество
4. лица будешь сравнивать скорее всего KNN, так как прочитал в статьях, что он показывает лучшее качество, но так же будешь пробовать другие и сравнивать качество

## реализация класса для поиска лиц на фото

```python
class AlignDlib:
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
	def getAllFaceBoundingBoxes(self, rgbImg):
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):

        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None
```

## поиск 1 лица на фото

```python

# Load an image of Jacques Chirac
jc_orig = load_image(metadata[2].image_path())

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
plt.figure(1, figsize=[8,8])
plt.subplot(121)
plt.imshow(jc_orig)

# Show original image with bounding box
plt.subplot(122)
plt.imshow(jc_orig)
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

```

<img src='./img1.png'>

## Пример поиска нескольких лиц на фото

```python

jc_orig = load_image(metadata[24].image_path())

# Detect face and return bounding box
bb = alignment.getAllFaceBoundingBoxes(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
# jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
plt.figure(1, figsize=[15,15])
plt.subplot(121)
plt.imshow(jc_orig)

# Show original image with bounding box
plt.subplot(122)
plt.imshow(jc_orig)
for el in bb:
    plt.gca().add_patch(patches.Rectangle((el.left(), el.top()), el.width(), el.height(), fill=False, color='red'))

```
<img src='./img2.png'>

