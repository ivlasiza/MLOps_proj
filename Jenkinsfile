pipeline {
    agent any

//     environment {
//         JENKINS_HOME = "$JENKINS_HOME"
//         BUILD = "${JENKINS_HOME}/workspace/mlops_final"
//     }

    stages {
         stage('Start') {
            steps {
                script {
                    echo 'Начало работы скриптов.'
                }
            }
        }
        stage('Preparation') {
            steps {
                // Очистка рабочего пространства
                cleanWs()
                checkout scm
            }
        }

        stage('Checkout') {
            steps {
                script {
                    // Получаем исходный код из репозитория Git
                    git branch: 'main', url: 'https://github.com/lastinm/mlops_final.git'
                }
            }
        }

        stage('Setup Virtual Environment') {
            steps {
                // Создание виртуального окружения
                script {
                    if (isUnix()) {
                        sh 'python -m venv venv'
                    } else {
                        bat 'python -m venv venv'
                    }
                }
            }
        }

        stage('Activate venv') {
            steps {
                // Активация виртуального окружения
                script {
                    if (isUnix()) {
                        sh './venv/scripts/activate.bat'
                    } else {
                        bat '.\\venv\\scripts\\activate.bat'
                    }
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                // установка зависимостей
                script {
                    if (isUnix()) {
                        sh 'pip install -r requirements.txt'
                    } else {
                        bat 'pip install -r requirements.txt'
                    }
                }
            }
        }

        stage('Create dataset Titanic') {
            steps {
                script {
                    // Создаем и обучаем модель
                    if (isUnix()) {
                        dir('src') {
                            // Можно выбрать хороший датасет или модифицированный (плохой) датасет
                            sh 'python make_dataset_titanic.py'
                            //sh 'python dataset_titanic_modifed.py'
                        }
                    } else {
                        dir('src') {
                            // Можно выбрать хороший датасет или модифицированный (плохой) датасет
                            bat 'python make_dataset_titanic.py'
                            //bat 'python dataset_titanic_modifed.py'
                        }
                    }
                }
            }
        }

        stage('Create model Titanic') {
            steps {
                script {
                    // Создаем и обучаем модель
                    if (isUnix()) {
                        dir('src') {
                            sh 'python model_titanic.py'
                        }
                    } else {
                        dir('src') {
                            bat 'python model_titanic.py'
                        }
                    }
                }
            }
        }

        stage('App tests') {
            steps {
                script {
                    if (isUnix()) {
                        sh 'pytest -v'
                    } else {
                        bat 'pytest -v'
                    }
                }
            }
        }

        stage('Build Docker image') {
            steps {
                 script {
                    // Для Линукс
                    if (isUnix()) {
                        sh 'docker build -t titanic-img .'
                    } else {
                        bat "docker build -t titanic-img -f Dockerfile ."
                    }
                 }
            }
        }

        stage('Finish') {
            steps {
                script {
                    echo 'Работа скриптов завершена успешно'
                }
            }
        }
    }
}
