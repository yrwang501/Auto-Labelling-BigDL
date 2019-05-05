from bayes_opt import BayesianOptimization
import urllib.request
import json
import time

def http_get_json(url):
    contents = urllib.request.urlopen("http://localhost:13346/"+url).read().decode("utf-8")
    return json.loads(contents)

def train(**kwargs):
    ret = http_get_json("train/0/1000/{}/{}/{}".format(kwargs['hue'],kwargs['contrast'],kwargs['learning_rate']))
    if not ret['accepted']:
        raise RuntimeError("task not accepted\n" + str(ret))
    while True:
        time.sleep(10)
        ret = http_get_json("status")
        if ret['status'] == 'idle':
            break
    return ret['accuracy']

pbounds = {
    #'contrast': (0.1, 2.0),
    #'brightness': (-100, 100),
    'contrast': (0.01, 2),
    'hue': (0, 179), 
    'learning_rate': (0.05, 1.5)
}

def get_optimizer():

    optimizer = BayesianOptimization(
        f=train,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={"contrast": 0.1, "hue": 0, "learning_rate": 0.12},
        lazy=True,
    )
    return optimizer

if __name__ == "__main__":
    get_optimizer().maximize(
            init_points=3,
            n_iter=1000,
    )
