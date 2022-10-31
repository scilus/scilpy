pipeline {
    agent any

    stages {
        stage('Build') {
            stages {
                stage('Python3.7') {
                    steps {
                        withPythonEnv('CPython-3.7') {
                            sh '''
                                pip3 install wheel==0.37.*
                                pip3 install setuptools==65.2.*
                                pip3 install numpy==1.21.*
                                pip3 install Cython==0.29.*
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
                        pip3 install wheel==0.37.*
                        pip3 install setuptools==65.2.*
                        pip3 install numpy==1.21.*
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
            script {
                if (env.CHANGE_ID) {
                    if (pullRequest.createdBy != "arnaudbore"){
                        pullRequest.createReviewRequests(['arnaudbore'])
                    }
                    else{
                        pullRequest.createReviewRequests(['GuillaumeTh'])
                    }
                }
            }
        }
        failure {
            script {
                // CHANGE_ID is set only for pull requests, so it is safe to access the pullRequest global variable
                if (env.CHANGE_ID) {
                    pullRequest.comment('Build Failed :boom: \n' + env.BUILD_URL)
                }
            }
        }
        success {
            script {
                // CHANGE_ID is set only for pull requests, so it is safe to access the pullRequest global variable
                if (env.CHANGE_ID) {
                    pullRequest.comment('Build passed ! Good Job :beers: !')
                }
            }
        }
    }
}
