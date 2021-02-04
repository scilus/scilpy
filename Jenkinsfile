pipeline {
    agent any

    stages {
        stage('Test') {
            agent { docker { image 'python:3.6-alpine' } }
            steps {
                sh '''
                    python setup.py develop
                    export MPLBACKEND="agg"
                    pytest -v
                   '''
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying.'
            }
        }
    }
}
