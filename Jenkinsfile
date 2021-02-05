pipeline {
    agent none

    stages {
        stage('Test') {
            agent {
                docker {
                      // Set both label and image
                      label 'docker'
                      image 'python:3.6-alpine'
                }
            }
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
