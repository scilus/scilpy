import numpy as np

orig_strl_additional_exit_point = np.array([[79.61719,  77.53906,   6.65625],
                                            [78.76172,  77.984375,   6.90625],
                                            [76.64453,  79.69531,   8.03125],
                                            [74.41992,  81.609375,   9.875],
                                            [73.259766,  82.52344,  11.15625],
                                            [72.63574,  83.828125,  13.71875],
                                            [72.84082,  84.453125,  16.625],
                                            [72.66113,  84.1875,  18.5625],
                                            [72.47656,  84.546875,  20.96875],
                                            [72.22461,  84.97656,  22.34375],
                                            [72.12207,  84.77344,  23.8125],
                                            [72.60156,  85.48828,  26.65625],
                                            [72.69336,  85.38281,  29.09375],
                                            [72.19629,  86.078125,  30.875],
                                            [71.10651,  87.33594,  32.6875],
                                            [70.11719,  88.19531,  34.765625],
                                            [69.953125,  87.640625,  36.640625],
                                            [70.09912,  85.875,  44.34375],
                                            [69.66406,  85.89453,  47.78125],
                                            [68.86328,  86.88672,  50.453125],
                                            [67.197266,  89.28906,  56.164062],
                                            [66.390625,  90.51953,  58.125],
                                            [65.15234,  91.98535,  59.695312],
                                            [64.14844,  93.18811,  61.597656],
                                            [62.648438,  94.493164,  64.41406],
                                            [61.921875,  95.25195,  66.06348],
                                            [61.28125,  95.08008,  68.94141],
                                            [60.882812,  95.42578,  72.34766],
                                            [59.210938,  97.69531,  79.75],
                                            [58.757812,  98.46484,  82.046875],
                                            [57.890625,  99.07422,  84.8125],
                                            [57.570312, 100.11328,  86.453125],
                                            [56.734375, 100.79297,  88.078125],
                                            [56.117188, 101.53906,  89.203125],
                                            [54.734375, 102.28906,  90.390625],
                                            [53.21875, 103.5625,  91.859375],
                                            [49.59375, 105.64844,  94.453125],
                                            [48.234375, 107.,  96.703125],
                                            [45.96875, 108.41406, 100.25],
                                            [45.796875, 110.140625, 103.25],
                                            [45.046875, 110.671875, 104.40625],
                                            [43.828125, 112.0625, 106.0625],
                                            [42.25, 113.0625, 106.59375],
                                            [40.21875, 113.84375, 105.40625],
                                            [37.65625, 113.078125, 104.125],
                                            [34.125, 112., 101.625],
                                            [32.28125, 112.328125, 101.03125],
                                            [27.9375, 112.296875, 102.03125],
                                            [26.28125, 112.921875, 102.84375],
                                            [23.90625, 113.5, 103.03125],
                                            [20.59375, 114.453125, 102.6875],
                                            [19., 115.34375, 103.40625]])
inter_vox_additional_exit_point = np.array([[79,  77,   6],
                                            [78,  77,   6],
                                            [78,  78,   6],
                                            [78,  78,   7],
                                            [77,  78,   7],
                                            [77,  79,   7],
                                            [76,  79,   7],
                                            [76,  79,   8],
                                            [76,  80,   8],
                                            [75,  80,   8],
                                            [75,  80,   9],
                                            [75,  81,   9],
                                            [74,  81,   9],
                                            [74,  81,  10],
                                            [73,  81,  10],
                                            [73,  82,  10],
                                            [73,  82,  11],
                                            [73,  82,  12],
                                            [73,  83,  12],
                                            [72,  83,  12],
                                            [72,  83,  13],
                                            [72,  83,  14],
                                            [72,  84,  14],
                                            [72,  84,  15],
                                            [72,  84,  16],
                                            [72,  84,  17],
                                            [72,  84,  18],
                                            [72,  84,  19],
                                            [72,  84,  20],
                                            [72,  84,  21],
                                            [72,  84,  22],
                                            [72,  84,  23],
                                            [72,  84,  24],
                                            [72,  85,  24],
                                            [72,  85,  25],
                                            [72,  85,  26],
                                            [72,  85,  27],
                                            [72,  85,  28],
                                            [72,  85,  29],
                                            [72,  85,  30],
                                            [72,  86,  30],
                                            [72,  86,  31],
                                            [71,  86,  31],
                                            [71,  86,  32],
                                            [71,  87,  32],
                                            [70,  87,  32],
                                            [70,  87,  33],
                                            [70,  87,  34],
                                            [70,  88,  34],
                                            [70,  88,  35],
                                            [70,  87,  35],
                                            [70,  87,  36],
                                            [69,  87,  36],
                                            [69,  87,  37],
                                            [69,  87,  38],
                                            [69,  87,  39],
                                            [70,  87,  39],
                                            [70,  86,  39],
                                            [70,  86,  40],
                                            [70,  86,  41],
                                            [70,  86,  42],
                                            [70,  86,  43],
                                            [70,  85,  43],
                                            [70,  85,  44],
                                            [70,  85,  45],
                                            [69,  85,  45],
                                            [69,  85,  46],
                                            [69,  85,  47],
                                            [69,  85,  48],
                                            [69,  86,  48],
                                            [69,  86,  49],
                                            [68,  86,  49],
                                            [68,  86,  50],
                                            [68,  87,  50],
                                            [68,  87,  51],
                                            [68,  87,  52],
                                            [68,  87,  53],
                                            [68,  88,  53],
                                            [67,  88,  53],
                                            [67,  88,  54],
                                            [67,  88,  55],
                                            [67,  89,  55],
                                            [67,  89,  56],
                                            [66,  89,  56],
                                            [66,  89,  57],
                                            [66,  90,  57],
                                            [66,  90,  58],
                                            [65,  90,  58],
                                            [65,  91,  58],
                                            [65,  91,  59],
                                            [65,  92,  59],
                                            [64,  92,  59],
                                            [64,  92,  60],
                                            [64,  92,  61],
                                            [64,  93,  61],
                                            [63,  93,  61],
                                            [63,  93,  62],
                                            [63,  93,  63],
                                            [63,  94,  63],
                                            [62,  94,  63],
                                            [62,  94,  64],
                                            [62,  94,  65],
                                            [62,  95,  65],
                                            [61,  95,  65],
                                            [61,  95,  66],
                                            [61,  95,  67],
                                            [61,  95,  68],
                                            [61,  95,  69],
                                            [61,  95,  70],
                                            [61,  95,  71],
                                            [60,  95,  71],
                                            [60,  95,  72],
                                            [60,  95,  73],
                                            [60,  95,  74],
                                            [60,  96,  74],
                                            [60,  96,  75],
                                            [60,  96,  76],
                                            [59,  96,  76],
                                            [59,  96,  77],
                                            [59,  97,  77],
                                            [59,  97,  78],
                                            [59,  97,  79],
                                            [59,  97,  80],
                                            [59,  98,  80],
                                            [58,  98,  80],
                                            [58,  98,  81],
                                            [58,  98,  82],
                                            [58,  98,  83],
                                            [58,  98,  84],
                                            [57,  98,  84],
                                            [57,  99,  84],
                                            [57,  99,  85],
                                            [57,  99,  86],
                                            [57, 100,  86],
                                            [57, 100,  87],
                                            [56, 100,  87],
                                            [56, 100,  88],
                                            [56, 101,  88],
                                            [56, 101,  89],
                                            [55, 101,  89],
                                            [55, 102,  89],
                                            [55, 102,  90],
                                            [54, 102,  90],
                                            [54, 102,  91],
                                            [53, 102,  91],
                                            [53, 103,  91],
                                            [53, 103,  92],
                                            [52, 103,  92],
                                            [52, 104,  92],
                                            [51, 104,  92],
                                            [51, 104,  93],
                                            [50, 104,  93],
                                            [50, 105,  93],
                                            [50, 105,  94],
                                            [49, 105,  94],
                                            [49, 105,  95],
                                            [49, 106,  95],
                                            [48, 106,  95],
                                            [48, 106,  96],
                                            [48, 107,  96],
                                            [48, 107,  97],
                                            [47, 107,  97],
                                            [47, 107,  98],
                                            [46, 107,  98],
                                            [46, 107,  99],
                                            [46, 108,  99],
                                            [46, 108, 100],
                                            [45, 108, 100],
                                            [45, 108, 101],
                                            [45, 109, 101],
                                            [45, 109, 102],
                                            [45, 109, 103],
                                            [45, 110, 103],
                                            [45, 110, 104],
                                            [44, 110, 104],
                                            [44, 111, 104],
                                            [44, 111, 105],
                                            [43, 111, 105],
                                            [43, 112, 105],
                                            [43, 112, 106],
                                            [42, 112, 106],
                                            [42, 113, 106],
                                            [41, 113, 106],
                                            [41, 113, 105],
                                            [40, 113, 105],
                                            [39, 113, 105],
                                            [39, 113, 104],
                                            [38, 113, 104],
                                            [37, 113, 104],
                                            [37, 113, 103],
                                            [37, 112, 103],
                                            [36, 112, 103],
                                            [36, 112, 102],
                                            [35, 112, 102],
                                            [34, 112, 102],
                                            [34, 112, 101],
                                            [34, 111, 101],
                                            [34, 112, 101],
                                            [33, 112, 101],
                                            [32, 112, 101],
                                            [31, 112, 101],
                                            [30, 112, 101],
                                            [29, 112, 101],
                                            [28, 112, 101],
                                            [28, 112, 102],
                                            [27, 112, 102],
                                            [26, 112, 102],
                                            [25, 112, 102],
                                            [25, 113, 102],
                                            [24, 113, 102],
                                            [24, 113, 103],
                                            [23, 113, 103],
                                            [23, 113, 102],
                                            [22, 113, 102],
                                            [22, 114, 102],
                                            [21, 114, 102],
                                            [20, 114, 102],
                                            [19, 114, 102],
                                            [19, 114, 103],
                                            [19, 115, 103],
                                            [18, 115, 103]])
in_vox_idx_additional_exit_point, out_vox_idx_additional_exit_point = 1, 195
points_to_indices_additional_exit_point = np.array([0,   1,   7,  12,  16,  20,  24,  26,  28,  30,  31,  35,  38,
                                                    40,  44,  48,  52,  63,  67,  72,  82,  86,  89,  94, 100, 104,
                                                    106, 111, 121, 126, 130, 133, 136, 138, 142, 145, 154, 158, 167,
                                                    172, 173, 179, 181, 184, 188, 196, 199, 205, 206, 211, 216, 220])
segment_additional_exit_point = np.array([[78.76172,  77.984375,   6.90625],
                                          [76.64453,  79.69531,   8.03125],
                                          [74.41992,  81.609375,   9.875],
                                          [73.259766,  82.52344,  11.15625],
                                          [72.63574,  83.828125,  13.71875],
                                          [72.84082,  84.453125,  16.625],
                                          [72.66113,  84.1875,  18.5625],
                                          [72.47656,  84.546875,  20.96875],
                                          [72.22461,  84.97656,  22.34375],
                                          [72.12207,  84.77344,  23.8125],
                                          [72.60156,  85.48828,  26.65625],
                                          [72.69336,  85.38281,  29.09375],
                                          [72.19629,  86.078125,  30.875],
                                          [71.10651,  87.33594,  32.6875],
                                          [70.11719,  88.19531,  34.765625],
                                          [69.953125,  87.640625,  36.640625],
                                          [70.09912,  85.875,  44.34375],
                                          [69.66406,  85.89453,  47.78125],
                                          [68.86328,  86.88672,  50.453125],
                                          [67.197266,  89.28906,  56.164062],
                                          [66.390625,  90.51953,  58.125],
                                          [65.15234,  91.98535,  59.695312],
                                          [64.14844,  93.18811,  61.597656],
                                          [62.648438,  94.493164,  64.41406],
                                          [61.921875,  95.25195,  66.06348],
                                          [61.28125,  95.08008,  68.94141],
                                          [60.882812,  95.42578,  72.34766],
                                          [59.210938,  97.69531,  79.75],
                                          [58.757812,  98.46484,  82.046875],
                                          [57.890625,  99.07422,  84.8125],
                                          [57.570312, 100.11328,  86.453125],
                                          [56.734375, 100.79297,  88.078125],
                                          [56.117188, 101.53906,  89.203125],
                                          [54.734375, 102.28906,  90.390625],
                                          [53.21875, 103.5625,  91.859375],
                                          [49.59375, 105.64844,  94.453125],
                                          [48.234375, 107.,  96.703125],
                                          [45.96875, 108.41406, 100.25],
                                          [45.796875, 110.140625, 103.25],
                                          [45.046875, 110.671875, 104.40625],
                                          [43.828125, 112.0625, 106.0625],
                                          [42.25, 113.0625, 106.59375],
                                          [40.21875, 113.84375, 105.40625],
                                          [37.65625, 113.078125, 104.125],
                                          [34.389843, 112.08086, 101.8125]])
