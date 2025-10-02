import websocket
import _thread
import time
import rel


from forecast.streaming_service import ForecastService


svc = ForecastService(model_dir="forecast/artifacts")

def on_message(ws, message):

    out = svc.process_message(message, 3000)
    if out["ready"]:
        ws.send(out["forecast"])

def on_error(ws, error):
    pass #print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8082/make_forecast",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever(dispatcher=rel, reconnect=5) 
    rel.signal(2, rel.abort)
    rel.dispatch()