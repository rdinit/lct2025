import websocket
import _thread
import time
import rel


from classification.streaming_service_clf import ClassificationService


csv =  ClassificationService(model_dir="classification/artifacts")

def on_message(ws, message):

    out = csv.process_message(message)
    if out["ready"]:
        ws.send(out["classification"])
    print(out)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://localhost:8082/make_classify",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever(dispatcher=rel, reconnect=5) 
    rel.signal(2, rel.abort)
    rel.dispatch()