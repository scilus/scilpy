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
                recordCoverage(
                    name: 'Scilpy Coverage Report',
                    sourceCodeRetention: 'MODIFIED',
                    tools: [[parser: 'COBERTURA',
                    pattern: '**/.test_reports/coverage.xml']])
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
