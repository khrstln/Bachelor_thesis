# Bachelor_thesis

## Описание 

Репозиторий содержит код на Matlab и Python, написанный в ходе работы над выпускной квалификационной работы на тему **"Восстановление дифференциальных уравнений, описывающих распространение излучения в сильно неоднородных рассеивающих средах, методами искусственного интеллекта"** на физическом факультете Университета ИТМО.

## Цель работы

Поиск одномерных дифференциальных уравнений, описывающих распространение излучения в сильно неоднородной рассеивающей среде на основании строгих численных решений уравнений Максвелла с использованием методов искусственного интеллекта

## Введение

Изучение распространения и рассеяния электромагнитных волн в сильно неоднородных рассеивающих средах находит свое применение в различных областях физики, где возникает задача определения состава среды на основании светорассеяния/спектроскопии и некоторых предположений о составе среды. Например, в биофизике для неинвазивной диагностики биологических тканей в инфракрасном и видимом диапазонах. В науках о Земле часто приходится работать с шероховатыми рассеивающими поверхностями, такими как почва, поверхность океанов, снежный покров и растительность, для чего применяются различные методы моделирования рассеяния света. 

Существует два основных подхода к моделированию распространения электромагнитного излучения в неоднородных средах. Первый подход основан на использовании методов численного решени уравнений Максвелла. Основным преимуществом такого подхода является возможность решения задачи рассеяния для частиц произвольной формы, но он может быть применен только для небольших групп частиц, так как используемые методы требуют достаточно большого количества вычислений. Второй подход основан на использовании известного из теории переноса излучения уравнения переноса излучения. $$\boldsymbol{s} \cdot \nabla I(\boldsymbol{r}, \boldsymbol{s}) = - (\sigma_{a}(\boldsymbol{r}) + \sigma_{s}(\boldsymbol{r})) I(\boldsymbol{r}, \boldsymbol{s}) + \frac{1}{4 \pi} \iint\limits_{\mathbb{S}^2} p(\boldsymbol{s}, \boldsymbol{s}') I(\boldsymbol{r}, \boldsymbol{s}') d \omega' + \mathcal{E}(\boldsymbol{r}, \boldsymbol{s}),$$ где $I(\boldsymbol{r}, \boldsymbol{s})$ - спектральная интенсивность излучения, $\boldsymbol{s}$ - направление, вдоль которого распространяется излучение, $\boldsymbol{r}$ - радиус вектор, $\sigma_{a}(\boldsymbol{r})$ - поперечное сечение поглощения, $\sigma_{s}(\boldsymbol{r})$ - поперечное сечение рассеяния, $p(\boldsymbol{s}, \boldsymbol{s}')$ - фазовая функция, равная доле излучения, распространяющегося в направлении $\boldsymbol{s}'$ и рассеянного в направлении $\boldsymbol{s}$, $\mathcal{E}(\boldsymbol{r}, \boldsymbol{s})$ - мощность излучения на единицу объема и на единицу телесного угла в направлении $\boldsymbol{s}$. Уравнение переноса излучения является интегро-дифференциальным и не имеет аналитического решения в общем случае, однако есть возможность получить простые модели распространения излучения в некоторых специальных случаях, что является основным преимуществом данного подхода. Но он применим только в случае достаточно низкой плотности рассеивающих частиц.

Таким образом, необходимы исследования более простых аппроксимаций сложной модели распространения излучения в сильно неоднородной среде. Основной целью работы является поиск одномерных дифференциальных уравнений, описывающих распространение излучения в сильно неоднородной рассеивающей среде, на основании строгих численных решений уравнений Максвелла с использованием методов искусственного интеллекта.

## Основная часть

### 1. Сбор данных

#### 1.1. Моделирование среды

Для моделирования двумерной неоднородной среды использовалась модель суперячейки с периодом $\Lambda$, конечной толщиной $H$, область между включениями заполнена свободным пространством $(\varepsilon_1 = 1)$. Верны следующие предельные соотношения: $\frac{H}{\lambda} \gg 1$, $\frac{\Lambda}{\lambda} \gg 1$, $\frac{l}{\lambda} \sim 1$, где $l$ - характерный линейный размер включений, $\lambda$ - длина падающей волны. Схема используемой модели приведена на рисунке ниже ![Supercell](https://github.com/khrstln/Bachelor_thesis/blob/development/images/Supercell_dif_permittivities.png) 

#### 1.2. Численное решение уравнений Максвелла

Для сбора данных необходимо было получить численные решения уравнений Максвелла внутри неоднородной подложки. Для этого подложка "разрезалась" на тонкие слои, как это показано на рисунке ниже ![Sliced supercell](images/Supercell_sliced.png) На каждом таком слое возникала задача дифракции, для решения которой в рамках модели суперячейки может быть использован Фурье-модальный метод ![One slice](images/Supercell_one_slice.png) Применяя Фурье-модальный метод на каждом тонком слое, можно рассчитать разложение поля в ряд Фурье на произвольном расстоянии от верхней границы подложки. В качестве данных, описывающих распространение энергии в неоднородной среде, использовалось значение нулевой гармоники Фурье проекции вектора Пойнтинга на вертикальную ось.

где $c^{\pm}_{0, \text{TE}}$ - значения амплитуд нулевых гармоник Фурье, распространяющихся в положительном и отрицательном направлениях оси $Oy$, рассчитанные с помощью Фурье-модального метода, $k_y$ - проекция волнового вектора на вертикальную ось $Oy$, $\omega$ - частота.

Моделирование среды и сбор данных производился с помощью языка программирования Matlab, так как существует реализация Фурье-модального метода на этом языке программирования.

### 3. Предобработка данных

Для последующего формирования дифференциальных уравнений необходимо численно рассчитать значения производных исследуемой функции по полученным на предыдущем шаге данным. Производные в ходе работы вычислялись с помощью численного дифференцирования многочлена Чебышева, построенного по исходным данным. Также в проекте предусмотрена возможность предварительного сглаживания исходных данных с помощью гауссова ядра, что может быть полезно в случае, когда уровень шума в данных достаточно высок. 

Для численного дифференцирования может быть применена аппроксимация исходных данных нейросетью и применение автоматического дифференцирования для вычисления прозводных. Такая возможность также предусмотрена в проекте. 

### 4. Восстановление дифференциальных уравнений

Для восстановления дифференциальных уравнений, описывающих исследуемый процесс, на основе данных использовался алгоритм, объединяющий генетический алгоритм и LASSO-регрессию фреймворк. Для определения оптимального набора слагаемых в уравнении использовался генетический алгоритм, для вычисления промежуточных значений коэффициентов и определения значимых из них применялась LASSO-регрессия. К полученной в результате работы алгоритма популяции уравнений применялась линейная регрессия для окончательного вычисления коэффициентов в уравнении. Описанный алгоритм реализован в фреймворке [EPDE](https://github.com/ITMO-NSS-team/EPDE/tree/main) для языка программирования Python.

### 5. Численное решение дифференциальных уравнений

Для выбора из популяции уравнений, решения которых наилучшим образом описывают исследуемый процесс, необходимо было применить методы их численного решения. Для получения таких решений использовались методы классического машинного обучения: искомая функция аппроксимировалась некоторой параметризованной моделью, затем решалась задача минимизации ошибки по набору параметров модели. Используемый алгоритм реализован в библиотеке [TEDEouS](https://github.com/ITMO-NSS-team/torch_DE_solver/tree/main) для языка программирования Python. Преимуществом использования данной библиотеки является совместимость с фреймворком [EPDE](https://github.com/ITMO-NSS-team/EPDE/tree/main), с помощью которого восстанавливались уравнения.

## Физические параметры исследуемой системы

В работе исследовалась двумерная подложка коненчной толщины с диэлектрическими включениями с одинаковыми значениями радиусов и диэлектрических проницаемостей, имеющая периодичность вдоль горизонтальной оси $Ox$ и расположенная в свободном пространстве. Падающая на подложку волна распространялась вдоль вертикальной оси $Oy$. Общая схема исследуемой системы представлена на рисунке ниже ![System scheme](https://github.com/khrstln/Bachelor_thesis/blob/development/images/System_scheme.png) Падающее на исследуемую структуру поле представляло собой TE-поляризованную плоскую монохроматическую волну, длина волны которой составляла $\lambda = 0.5 \text{ мкм}$. Полная толщина $H = 50 \text{ мкм}$ ($\frac{H}{\lambda} = 100$ в безразмерных единицах), период $\Lambda = 15 \text{ мкм}$ ($\frac{\Lambda}{\lambda} = 30$ в безразмерных единицах). Область между цилиндрическими включениями заполнена свободным пространством с диэлектрической проницаемостью $\varepsilon_{1} = 1$. Диэлектрическая проницаемость включений составляет $\varepsilon_{2} = 4$. Плотность упаковки включений составляла $n = \widetilde{n} \cdot \pi r_{0}^{2} = 0.3$, где $\widetilde{n}$ - концентрация рассеивающих частиц. Радиусы цилиндров были одинаковыми в ходе каждого запуска, в общем случае включения внутри подложки могут иметь разные радиусы. На этапе расчета поля внутри подложки с помощью Фурье-модального метода учитывалось $N = 150$ гармоник Фурье.

## Пример результатов

В ходе исследования выяснилось, что следующие значения параметров для алгоритмов восстановления дифференциальных уравнений по данным и их решения дают наиболее приемлемые с точки зрения физики и значения ошибки на тестовой выборке результаты
* ```pop_size = 6``` - размер популяции в алгоритме восстановления уравнений
* ```factors_max_number = 1``` - максимальное количество множителей в слагаемом дифференциального уравнения
* ```poly_order = 4``` - максимальный порядок многочлена от искомой функции
* ```max_deriv_order = 2``` - максимальный порядок производной в уравнении
* ```training_epde_epochs = 100``` - количество эпох в алгоритме восстановления дифференциальных уравнений
* ```training_tedeous_epochs = 10000``` - количество эпох в алгоритме решения дифференциального уравнения

Уравнение, восстановленное по данным, для значения радиуса $r_0 = 0,1 \text{ мкм}$

$$-4,033 \cdot I - 3,063 \cdot I^4 - 0,001  = \frac{dI}{dy}$$

Уравнение имеет граничное условие $I_{0, y}(0) = -1$. График численного решения и тестовые данные, а также ошибка на тестовой выборке приведены на рисунке ниже

![Solution](https://github.com/khrstln/Bachelor_thesis/blob/development/images/sln_0.1_0_0.png)

