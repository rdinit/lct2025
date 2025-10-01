from forecast.streaming_service import ForecastService

svc = ForecastService(model_dir="forecast/artifacts")

msg = ""

for i in range(15):
    msg += "1609459200,85.5,12.3\n"


K = 5 # прогноз на 5 шагов вперёд


out = svc.process_message(msg, K)

if out["ready"] == True:
    print(out["forecast"])
    pass
else:
    print(f"Need {out["needed"]} more points before forecasting can start")  