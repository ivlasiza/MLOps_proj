# MLOps_proj
Структура проекта и назначение файлов:

src/                                  - каталог с исполняемыми скриптами приложения
    src/api_app_titanic.py            - API приложение на FASTAPI
    src/dataset_titanic_modifed.py    - создаем "плохой" датасет
    src/make_dataset_titanic.py       - сохраняем датасет Titanic
    src/model_titanic.py              - обучаем модель на данных датасета Titanic

test/
    test/api_app_test.py              - модульное тестирование API приложения
    test/dataset_test.py              - функциональное тестирование предсказательной способности модели

./doker_container_run.bat             - запуск docker контейнера с приложением из образа (Windows)
./doker_container_run.sh              - запуск docker контейнера с приложением из образа (Unix)
./Dokerfile                           - файл для создания doker образа приложения
./Jenkinsfile                         - конвейер для автоматического развертывания приложения, 
                                        тестирования и сборки doker образа
./predict_titanic.bat                 - командный файл для проверки работы API приложения (Windows)
./predict_titanic.sh                  - командный файл для проверки работы API приложения (Unix)

Приложение собирается в docker контейнере. Сборка приложения, тестирование и создание docker образа происходит в
автоматическом режиме в Jenkins на основе файла сценария Jenkinsfile.

После создания docker образа приложения запускаем командный файл doker_container_run.bat (doker_container_run.sh) для 
сборки и запуска docker контейнера с приложением. Приложение запускается на порту 8091.

Для проверки работы предложения запускаем командные файлы: predict_titanic.bat (predict_titanic.sh)
