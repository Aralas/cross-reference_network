# -*- coding:utf-8 -*-
"""
@author:Xu Jingyi
@file:main.py.py
@time:2019-03-0413:28
"""
import psutil
import os
import RunTest6

file_index = 1

RunTest6.initialization(1)

info = psutil.virtual_memory()
print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
print(u'总内存：', info.total)
print(u'内存占比：', info.percent)
print(u'cpu个数：', psutil.cpu_count())

for key in list(globals().keys()):
    if not key.startswith("__"):
        globals().pop(key)

import psutil
import os

info = psutil.virtual_memory()
print(u'清除所有变量')
print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
print(u'总内存：', info.total)
print(u'内存占比：', info.percent)
print(u'cpu个数：', psutil.cpu_count())

for block in range(11):
    import RunTest6
    file_index = 1

    start_section = 5 * (block + 1)
    end_section = 5 * (block + 2)
    RunTest6.run_cross_reference(start_section, end_section, file_index)

    info = psutil.virtual_memory()
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：', info.total)
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())

    for key in list(globals().keys()):
        if not key.startswith("__"):
            globals().pop(key)

    import psutil
    import os

    info = psutil.virtual_memory()
    print(u'清除所有变量')
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：', info.total)
    print(u'内存占比：', info.percent)
    print(u'cpu个数：', psutil.cpu_count())