import websocket
import _thread
import time
import rel
from anomaly_detection.anomaly_norm import MADAnomalyDetector
import json

det = MADAnomalyDetector()

def on_message(ws, message):
    current_gid = "0"
    current_sid = "0"

    data = []
    x = []
    y = []
    cols = message.strip().split('\n')

    for col in cols:
        parts = col.split(',')
        x.append(float(parts[1]))
        y.append(float(parts[2]))
        data.append({
            "ts": float(parts[0]),
            "bpm": float(parts[1]),
            "ut": float(parts[2]),
            "gid": current_gid,
            "sid": current_sid
        })


    outBpm = det.detect_anomalies(x)
    outUterus = det.detect_anomalies(y)

    bpmTimestamps = []
    uterusTimestamps = []
    for i in outBpm:
        bpmTimestamps.append(data[i]["ts"])

    for i in outUterus:
        uterusTimestamps.append(data[i]["ts"])

    ans = {
        "bpm": json.dumps(bpmTimestamps),
        "uterus": json.dumps(uterusTimestamps)
    }
    ws.send(ans)

def on_error(ws, error):
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8082/make_anomaly",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever(dispatcher=rel, reconnect=5) 
    rel.signal(2, rel.abort)
    rel.dispatch()