pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                withPythonEnv('CPython-3.6') {
                        sh '''
                            apt-get install python3-venv
                            python setup.py develop
                            export MPLBACKEND="agg"
                            pytest -v
                           '''
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
