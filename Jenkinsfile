pipeline {
    agent any

    stages {
        stage('Test') {
            parallel {
                stage('Python3.6') {
                    steps {
                        withPythonEnv('CPython-3.6') {
                            sh '''
                                pip3 install numpy==1.18.* wheel
                                pip3 install -e .
                                export MPLBACKEND="agg"
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

                            '''
                        }
                    }
                }
                stage('Python3.8') {
                    steps {
                        withPythonEnv('CPython-3.8') {
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
            when {
                branch 'master'
            }
            steps {
                echo 'Deploying.'
            }
        }
    }
}
