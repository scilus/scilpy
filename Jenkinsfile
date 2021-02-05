pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                sh 'apt-get install python3-venv'
                withPythonEnv('CPython-3.6') {
                        sh 'python --version'
                }
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying.'
            }
        }
    }
}
