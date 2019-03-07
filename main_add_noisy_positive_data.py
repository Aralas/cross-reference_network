# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:main.py.py
@time:2019-03-0413:28
"""

import RunTest7
import numpy as np

file_index = 1
lambda_weight = np.zeros(200)

RunTest7.initialization(file_index, lambda_weight)

for section in range(200):
    import RunTest7
    import numpy as np

    file_index = 1
    lambda_weight = np.zeros(200)

    RunTest7.run_cross_reference(section, file_index, lambda_weight)

    for key in list(globals().keys()):
        if not key.startswith("__"):
            if key != 'section':
                globals().pop(key)

