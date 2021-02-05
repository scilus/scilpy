pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                sh 'sudo apt-get install -y python3-venv build-essential libblas-dev liblapack-dev python3-tk python3-dev'
                withPythonEnv('CPython-3.6') {
                        sh '''
                           pip3 install numpy==1.18.*
                           pip3 install -e .
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
