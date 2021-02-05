pipeline {
    agent any

    stages {
        stage('Install dependencies') {
            steps {
                sh 'sudo apt-get install -y python3-venv build-essential libblas-dev liblapack-dev python3-tk python3-dev cmake'
            }
        }

        stage('Test') {
            parallel {
                stage('Python3.6') {
                    agent { docker { image: "python:3.6-alpine" }}
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
