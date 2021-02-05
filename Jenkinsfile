pipeline {
    agent any

    stages {
        stage('Test') {
            parallel {
                stage('Python3.6') {
                    agent {
                            docker { image 'python:3.6-alpine' }
                        }
                    steps {
                           sh '''
                               pip3 install numpy==1.18.* wheel
                               pip3 install -e .
                               export MPLBACKEND="agg"
                           '''
                    }
                }
                stage('Python3.7') {
                    steps {
                        withPythonEnv('CPython-3.7') {
                            sh '''
                                pip3 install numpy==1.18.* wheel
                                pip3 install -e .
                                export MPLBACKEND="agg"

                            '''
                        }
                    }
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
