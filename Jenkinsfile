pipeline {
    agent none

    stages {
        stage('Test') {
            steps {
                agent { docker { image 'python:3.6' } }
                sh '''
                    python setup.py develop
                    export MPLBACKEND="agg"
                    pytest -v
                   '''
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}
