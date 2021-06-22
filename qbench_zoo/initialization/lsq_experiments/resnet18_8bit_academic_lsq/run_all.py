Observer_dict = ['MinMaxObserver',
                 'ClipStdObserver',
                 'AverageMinMaxObserver',
                 'LSQObserver',
                 'LSQPlusObserver',
                 'AverageLSQObserver',
                 'QuantileMovingAverageObserver',
                 'MSEObserver',
                 'KLDObserver']

observer = []
for a in Observer_dict:
    for w in Observer_dict:
        observer.append((a, w))

import os
for a, w in observer:
    os.system('bash run.sh {} {}'.format(a, w))
