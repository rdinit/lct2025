from forecast.streaming_service import ForecastService

from anomaly_detection.streaming_anomaly_service import AnomalyService


svc = ForecastService(model_dir="forecast/artifacts")

det = AnomalyService(model_dir="anomaly_detection/artifacts")

msg = ""

for i in range(25):
    det.process_message("0,85.5,12.3")
    msg += "0,85.5,12.3\n"


K = 5 # прогноз на 5 шагов вперёд


out = svc.process_message(msg, K)

if out["ready"] == True:
    print(out["forecast"])
    pass
else:
    print(f"Need {out["needed"]} more points before forecasting can start")  

out = det.process_message("0,85.5,12.3")

if out["ready"]:
     # timestamp,group_id,sequence_id,bpm,uterus,bpm_anomaly,uterus_anomaly,anomaly_score
    print(out["detection"])
else:
    print(out["needed"])