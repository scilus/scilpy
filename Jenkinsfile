pipeline {
    agent any

    stages {
        stage('Test') {
            parallel {
                stage('Python3.6') {
                    steps {
                        withPythonEnv('CPython-3.6') {
                            sh '''
                                export SCILPY_HOME=$PWD
                                pip3 install numpy==1.18.* wheel
                                pip3 install -e .
                                export MPLBACKEND="agg"
                                pytest -v
                            '''
                        }
                    }
                }
                stage('Python3.7') {
                    steps {
                        withPythonEnv('CPython-3.7') {
                            sh '''
                                export SCILPY_HOME=$PWD
                                pip3 install numpy==1.18.* wheel
                                pip3 install -e .
                                export MPLBACKEND="agg"
                                pytest -v
                            '''
                        }
                    }
                }
            }
        }

        stage('Deploy') {
            when {
                branch 'master'
            }
            steps {
                echo 'Deploying.'
            }
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}
