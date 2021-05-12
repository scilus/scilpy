pipeline {
    agent any

    stages {
        stage('Build') {
            stages {
                stage('Python3.6') {
                    steps {
                        withPythonEnv('CPython-3.6') {
                            sh '''
                                pip3 install numpy==1.18.* wheel
                                pip3 install -e .
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
                            '''
                        }
                    }
                }
            }
        }

        stage('Test') {
            steps {
                withPythonEnv('CPython-3.7') {
                    sh '''
                        pip3 install numpy==1.18.* wheel
                        pip3 install -e .
                        export MPLBACKEND="agg"
                        export OPENBLAS_NUM_THREADS=1
                        pytest -v
                    '''
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
            script {
                pullRequest.createReviewRequests(['arnaudbore'])
            }
        }
        failure {
            script {
                // CHANGE_ID is set only for pull requests, so it is safe to access the pullRequest global variable
                if (env.CHANGE_ID) {
                    pullRequest.comment('Build Failed :' + env.BUILD_URL)
                }
            }
        }
    }
}
