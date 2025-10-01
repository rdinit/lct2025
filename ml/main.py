from forecast.streaming_service import ForecastService

from anomaly_detection.streaming_anomaly_service import AnomalyService

# from classification.streaming_service_clf import ClassificationService


svc = ForecastService(model_dir="forecast/artifacts")

det = AnomalyService(model_dir="anomaly_detection/artifacts")


# csv = ClassificationService()
import time

def current_milli_time():
    return round(time.time() * 1000)
msg = ""

for i in range(6000):
    det.process_message(f"{i},85.5,12.3")
    msg += f"{i},85.5,12.3\n"


K = 3000 # прогноз на 3000 шагов вперёд



now = current_milli_time()
out = svc.process_message(msg, K)

if out["ready"] == True:

    print(current_milli_time() - now)
    print(len(out["forecast"].split('\n')))
    pass
else:
    print(f"Need {out["needed"]} more points before forecasting can start")  


now = current_milli_time()
out = det.process_message("0,85.5,12.3")

if out["ready"]:

    print(current_milli_time() - now)
     # timestamp,group_id,sequence_id,bpm,uterus,bpm_anomaly,uterus_anomaly,anomaly_score
    print(out["detection"])
else:
    print(out["needed"])