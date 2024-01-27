pipeline {
    options {
        disableConcurrentBuilds(abortPrevious: true)
        throttleJobProperty(categories: ['ci_all_builds'],
                            throttleEnabled: true,
                            throttleOption: 'category')
    }

    agent any

    stages {
        stage('Build') {
            stages {
                stage('Python3.10') {
                    steps {
                        withPythonEnv('CPython-3.10') {
                            sh '''
                                pip3 install wheel==0.38.*
                                pip3 install numpy==1.23.*
                                pip3 install Cython==0.29.*
                                pip3 install packaging==23.*
                                pip3 install -e .
                            '''
                        }
                    }
                }
            }
        }

        stage('Test') {
            environment {
                CODECOV_TOKEN = credentials('scilpy-codecov-token')
            }
            steps {
                withPythonEnv('CPython-3.10') {
                    sh '''
                        pip3 install pytest-cov pytest-html
                        pip3 install wheel==0.38.*
                        pip3 install numpy==1.23.*
                        pip3 install packaging==23.*
                        pip3 install -e .
                        export MPLBACKEND="agg"
                        export OPENBLAS_NUM_THREADS=1
                        pytest --cov-report term-missing:skip-covered
                    '''
                }
                discoverGitReferenceBuild()
                sh '''
                    curl https://keybase.io/codecovsecurity/pgp_keys.asc | gpg --no-default-keyring --import # One-time step
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    curl -Os https://uploader.codecov.io/latest/linux/codecov.SHA256SUM
                    curl -Os https://uploader.codecov.io/latest/linux/codecov.SHA256SUM.sig

                    gpg --verify codecov.SHA256SUM.sig codecov.SHA256SUM
                    shasum -a 256 -c codecov.SHA256SUM

                    chmod +x codecov
                    ./codecov -t ${CODECOV_TOKEN} \
                        -f .test_reports/coverage.xml \
                        -C ${GIT_PREVIOUS_COMMIT}
                '''
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
                        pullRequest.createReviewRequests(['frheault'])
                    }
                }
                xunit(
                    checksName: '',
                    tools: [JUnit(excludesPattern: '', failIfNotNew: false,
                            pattern: '**/.test_reports/junit.xml',
                            skipNoTestFiles: true,
                            stopProcessingIfError: true)]
                )
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
