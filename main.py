# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:main.py.py
@time:2019-03-0413:28
"""
import psutil
import os
import RunTest6
import numpy as np

file_index = 1
lambda_weight = np.zeros(60)

RunTest6.initialization(file_index, lambda_weight)

for section in range(60):
    import RunTest6
    import numpy as np

    file_index = 1
    lambda_weight = np.zeros(60)

    RunTest6.run_cross_reference(section, file_index, lambda_weight)

    info = psutil.virtual_memory()
    print('Memory used: ', psutil.Process(os.getpid()).memory_info().rss)
    print('Total memory: ', info.total)
    print('Memory percent: ', info.percent)
    print('cpu: ', psutil.cpu_count())

    for key in list(globals().keys()):
        if not key.startswith("__"):
            if key != 'section':
                globals().pop(key)

    import psutil
    import os

    info = psutil.virtual_memory()
    print('Clear all the variables')
    print('Memory used: ', psutil.Process(os.getpid()).memory_info().rss)
    print('Total memory: ', info.total)
    print('Memory percent: ', info.percent)
    print('cpu: ', psutil.cpu_count())
