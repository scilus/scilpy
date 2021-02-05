pipeline {
    agent any

    stages {
        stage('Test') {
            withPythonEnv('CPython-3.6') {
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
