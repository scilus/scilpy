pipeline {
    agent any

    stages {
        stage('Test') {
            parallel {
                steps {
                    sh 'sudo apt-get install -y python3-venv build-essential libblas-dev liblapack-dev python3-tk python3-dev cmake'
                }
                stage('Python3.6') {
                    steps {
                        withPythonEnv('CPython-3.6') {
                            sh '''
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
            steps {
                echo 'Deploying.'
            }
        }
    }
}
