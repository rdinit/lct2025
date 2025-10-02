import websocket
import _thread
import time
import rel
import json

import diagnoses.diagnoses_service as svc


def on_message(ws, message):

    current_gid = "0"
    current_sid = "0"

    x = []
    t = []
    cols = message.strip().split("\n")

    for col in cols:
        parts = col.split(",")
        x.append(float(parts[1]))
        t.append(float(parts[0]))

    data = {
        "tachycardia": svc.detect_tachycardia(x, t, svc.config),
        "bradycardia": svc.detect_bradycardia(x, t, svc.config),
        "decelerations": svc.detect_decelerations(x, t, svc.config),
        "reduced_variability": svc.detect_reduced_variability(x, t, svc.config),
    }

    ws.send(json.dumps(data))


def on_error(ws, error):
    pass  # print(error)


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    print("Opened connection")


if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(
        "ws://localhost:8082/back2ml?model=diagnoses",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever(dispatcher=rel, reconnect=5)
    rel.signal(2, rel.abort)
    rel.dispatch()
