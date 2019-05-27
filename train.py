from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger,ScreenLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import urllib.request
import requests 
from urllib import request, parse
import json
import time
import subprocess
import sys
import os

log_stderr = open("server_stderr.txt", "w")
log_stdout = open("server_stdout.txt", "w")
BASEURL="http://localhost:13346/"


class BasicObserver:
    def __init__(self, path):
        self.jsonlogger=JSONLogger(path="./logs.json")
        self.screenlogger=ScreenLogger()
    def update(self, event, instance):
        self.jsonlogger.update(event,instance)
        self.screenlogger.update(event,instance)

def http_get_json(url):
    with urllib.request.urlopen(BASEURL+url) as resp:
        contents = resp.read()
        return json.loads(contents)

def kill_server(the_process):
    try:
        http_get_json("stop")
    except:
        pass
    the_process.wait()   

def start_server():
    cmd='''$SPARK_HOME/bin/spark-submit \
    --master "spark://ai-master-bigdl-0.sh.intel.com:7077" \
    --driver-memory 4g --class com.intel.aimaster.TrainServer \
    --conf 'spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn'\
    --conf 'spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn' \
	--total-executor-cores 48\
	--executor-cores 24\
        --executor-memory 40g\
    target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    new-core-site.xml \
    new-hbase-site.xml \
    kfb_data \
    1 \
    40000 \
    144 \
    0.04 \
    5'''

    ret = subprocess.Popen(cmd, shell=True, stdout=log_stdout, stderr=log_stderr)
    while True:
        try:
            time.sleep(10)
            data = http_get_json("status")
            if data['status'] == 'idle':
                break
        except Exception as e:
            print(e)
            print("Server not ready, waiting")
    return ret

def get_full_model(kwargs):
    process = start_server()
    kwargs['rotations'] = round(float(kwargs['rotations']))
    
    rawret = requests.get(url = BASEURL+"getmodel", params = kwargs)
    print(rawret)
    ret =rawret.json()
    if not ret['accepted']:
        raise RuntimeError("task not accepted\n" + str(ret))
    while True:
        time.sleep(10)
        ret = http_get_json("status")
        if ret['status'] == 'idle':
            break
    kill_server(process)
    return ret['accuracy']

def train(**kwargs):
    process = start_server()
    rotations = round(float(kwargs['rotations']))
    ret = http_get_json("train/0/1000/{}/{}/{}/{}"
        .format(kwargs['hue'],kwargs['contrast'],kwargs['lr'], rotations))
    if not ret['accepted']:
        raise RuntimeError("task not accepted\n" + str(ret))
    while True:
        time.sleep(10)
        ret = http_get_json("status")
        if ret['status'] == 'idle':
            break
    kill_server(process)
    return ret['accuracy']

pbounds = {
    #'contrast': (0.1, 2.0),
    #'brightness': (-100, 100),
    'contrast': (0.0001, 200),
    'hue': (0, 179), 
    'lr': (0.05, 1.5),
    'rotations': (-0.49, 4.49)
}

def get_optimizer():

    optimizer = BayesianOptimization(
        f=train,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    if os.path.isfile('./oldlogs.json'):
        load_logs(optimizer, logs=["./oldlogs.json"])
        print("New optimizer is now aware of {} points.".format(len(optimizer.space)))

    else:
        optimizer.probe(
            params={"contrast": 0.0001, "hue": 0, "lr": 0.12, "rotations": 0},
            lazy=True,
        )
    logger = BasicObserver("./logs.json")
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)
    optimizer.subscribe(Events.OPTMIZATION_START,logger.screenlogger)
    optimizer.subscribe(Events.OPTMIZATION_END,logger.screenlogger)
    return optimizer

#hue=35.49,contrast=106.7,lr=0.05,rotations=4,modelpath=./tuned.obj,epochs=1
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Bad args")
    else:
        if sys.argv[1] == '-t' or sys.argv[1] == '--train':
            get_optimizer().maximize(
                    init_points=3,
                    n_iter=1000,
            )
        elif sys.argv[1] == '-m' or sys.argv[1] == '--full-model':
            if len(sys.argv) != 3:
                print("Bad args, expecting parameters")
            v = sys.argv[2].split(',')
            kwargs = dict()
            for arg in v:
                kv = arg.split('=')
                kwargs[kv[0]] = kv[1]
            print("Final accuracy", get_full_model(kwargs))
        else:
            print("Bad parameter ", sys.argv[1])
