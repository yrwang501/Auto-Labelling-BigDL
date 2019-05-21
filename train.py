from bayes_opt import BayesianOptimization
import urllib.request
import json
import time
import subprocess

log_stderr = open("server_stderr.txt", "w")
log_stdout = open("server_stdout.txt", "w")

def http_get_json(url):
    contents = urllib.request.urlopen("http://localhost:13346/"+url).read()
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
	--total-executor-cores 72\
	--executor-cores 24\
        --executor-memory 40g\
    target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
    new-core-site.xml \
    new-hbase-site.xml \
    kfb_data \
    1 \
    40000 \
    144 \
    0.03 \
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
    'rotations': (-0.51, 4.49)
}

def get_optimizer():

    optimizer = BayesianOptimization(
        f=train,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={"contrast": 0.0001, "hue": 0, "lr": 0.12, "rotations": 0},
        lazy=True,
    )
    return optimizer

if __name__ == "__main__":
    get_optimizer().maximize(
            init_points=3,
            n_iter=1000,
    )
