pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
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
