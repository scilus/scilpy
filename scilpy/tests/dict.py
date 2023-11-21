
dict_to_average = {
    "sub-01": {
        "AF_L": {'length': [1.00, 2.00]},
        "AF_R": {'length': [8.00, 2.00]},
    },
    "sub-02": {
        "AF_L": {'length': [9.00, 2.00]},
        "AF_R": {'length': [13.00, 2.00]},
    }
}

expected_dict_averaged = {
    "sub-01": {
        "AF_L": {'length': {'mean': 1.5, 'std': 0.5}},
        "AF_R": {'length': {'mean': 5.0, 'std': 3.0}},
    },
    "sub-02": {
        "AF_L": {'length': {'mean': 5.5, 'std': 3.5}},
        "AF_R": {'length': {'mean': 7.5, 'std': 5.5}},
    }
}


dict_1 = {
    "sub-01": {
        "AF_L": {
            "std_length": 1.00,
            "min_length": 2.00,
            "max_length": 3.00,
            "mean_length": 4.00
        },
        "AF_R": {
            "std_length": 5.00,
            "min_length": 6.00,
            "max_length": 7.00,
            "mean_length": 8.00
        }
    },
    "sub-02": {
        "AF_L": {
            "std_length": 9.00,
            "min_length": 10.00,
            "max_length": 11.00,
            "mean_length": 12.00
        },
        "AF_R": {
            "std_length": 13.00,
            "min_length": 14.00,
            "max_length": 15.00,
            "mean_length": 16.00
        }
    }
}

dict_2 = {
    "sub-01": {
        "AF_L": {
            "std_length": 17.00,
            "min_length": 18.00,
            "max_length": 19.00,
            "mean_length": 20.00
        },
        "AF_R": {
            "std_length": 21.00,
            "min_length": 22.00,
            "max_length": 23.00,
            "mean_length": 24.00
        }
    },
    "sub-02": {
        "AF_L": {
            "std_length": 25.00,
            "min_length": 26.00,
            "max_length": 27.00,
            "mean_length": 28.00
        },
        "AF_R": {
            "std_length": 29.00,
            "min_length": 30.00,
            "max_length": 31.00,
            "mean_length": 32.00
        }
    },
    "sub-03": {
        "AF_L": {
            "std_length": 33.00,
            "min_length": 34.00,
            "max_length": 35.00,
            "mean_length": 36.00
        },
        "LF_R": {
            "std_length": 37.00,
            "min_length": 38.00,
            "max_length": 39.00,
            "mean_length": 40.00
        }
    }
}


expected_merged_dict_12 = {
    'sub-01': [{'AF_L': {'std_length': 1.0,
                         'min_length': 2.0,
                         'max_length': 3.0,
                         'mean_length': 4.0},
                'AF_R': {'std_length': 5.0,
                         'min_length': 6.0,
                         'max_length': 7.0,
                         'mean_length': 8.0}},
               {'AF_L': {'std_length': 17.0,
                         'min_length': 18.0,
                         'max_length': 19.0,
                         'mean_length': 20.0},
                'AF_R': {'std_length': 21.0,
                         'min_length': 22.0,
                         'max_length': 23.0,
                         'mean_length': 24.0}}],
    'sub-02': [{'AF_L': {'std_length': 9.0,
                         'min_length': 10.0,
                         'max_length': 11.0,
                         'mean_length': 12.0},
                'AF_R': {'std_length': 13.0,
                         'min_length': 14.0,
                         'max_length': 15.0,
                         'mean_length': 16.0}},
               {'AF_L': {'std_length': 25.0,
                         'min_length': 26.0,
                         'max_length': 27.0,
                         'mean_length': 28.0},
                'AF_R': {'std_length': 29.0,
                         'min_length': 30.0,
                         'max_length': 31.0,
                         'mean_length': 32.0}}],
    'sub-03': {'AF_L': {'std_length': 33.0,
                        'min_length': 34.0,
                        'max_length': 35.0,
                        'mean_length': 36.0},
               'LF_R': {'std_length': 37.0,
                        'min_length': 38.0,
                        'max_length': 39.0,
                        'mean_length': 40.0}}}

expected_merged_dict_12_all_true = {
    'sub-01': {
        'AF_L': {
            'std_length': 18.0,
            'min_length': 20.0,
            'max_length': 22.0,
            'mean_length': 24.0},
        'AF_R': {
            'std_length': 26.0,
            'min_length': 28.0,
            'max_length': 30.0,
            'mean_length': 32.0}
    },
    'sub-02': {
        'AF_L': {
            'std_length': 34.0,
            'min_length': 36.0,
            'max_length': 38.0,
            'mean_length': 40.0},
        'AF_R': {
            'std_length': 42.0,
            'min_length': 44.0,
            'max_length': 46.0,
            'mean_length': 48.0}
    },
    'sub-03': {
        'AF_L': {
            'std_length': 33.0,
            'min_length': 34.0,
            'max_length': 35.0,
            'mean_length': 36.0},
        'LF_R': {
            'std_length': 37.0,
            'min_length': 38.0,
            'max_length': 39.0,
            'mean_length': 40.0}
    }
}

expected_merged_dict_12_recursive = {
    'sub-01': {'AF_L': {'std_length': [1.0, 17.0],
                        'min_length': [2.0, 18.0],
                        'max_length': [3.0, 19.0],
                        'mean_length': [4.0, 20.0]},
               'AF_R': {'std_length': [5.0, 21.0],
                        'min_length': [6.0, 22.0],
                        'max_length': [7.0, 23.0],
                        'mean_length': [8.0, 24.0]}},
    'sub-02': {'AF_L': {'std_length': [9.0, 25.0],
                        'min_length': [10.0, 26.0],
                        'max_length': [11.0, 27.0],
                        'mean_length': [12.0, 28.0]},
               'AF_R': {'std_length': [13.0, 29.0],
                        'min_length': [14.0, 30.0],
                        'max_length': [15.0, 31.0],
                        'mean_length': [16.0, 32.0]}},
    'sub-03': {'AF_L': {'std_length': 33.0,
                        'min_length': 34.0,
                        'max_length': 35.0,
                        'mean_length': 36.0},
               'LF_R': {'std_length': 37.0,
                        'min_length': 38.0,
                        'max_length': 39.0,
                        'mean_length': 40.0}}}
