import { useEffect, useRef } from "react";


export default function DataSender () {
    const uterusSender = useRef<WebSocket>(null!);

    const bpmSender = useRef<WebSocket>(null!);

    useEffect(() => {
        uterusSender.current = new WebSocket("ws://localhost:8082/send?sensor_id=uterus");

        uterusSender.current.onopen = () => {
            console.log("Opened UterusSender");
        };
        uterusSender.current.onclose = () => {
            console.log("Closed UterusSender");
        };
        uterusSender.current.onerror = () => {
            console.log("Error UterusSender");
        };
        const startTime = Date.now();


        function sendData () {
            const currentTime = (Date.now() - startTime) / 1000;

            const messageTime = currentTime;
            let randomValue = Math.random() * 100;
            let message = `${messageTime.toFixed(4)},${randomValue.toFixed(4)}`;

            if (uterusSender.current.readyState === WebSocket.OPEN) {
                uterusSender.current.send(message);
            }
            randomValue = Math.random() * 100;
            message = `${messageTime.toFixed(4)},${randomValue.toFixed(4)}`;
            if (bpmSender.current.readyState === WebSocket.OPEN) {
                bpmSender.current.send(message);
            }
        }
        const sendDataId = setInterval(sendData, 1000);
        bpmSender.current = new WebSocket("ws://localhost:8082/send?sensor_id=bpm");

        bpmSender.current.onopen = () => {
            console.log("Opened BpmSender");
        };
        bpmSender.current.onclose = () => {
            console.log("Closed BpmSender");
        };
        bpmSender.current.onerror = () => {
            console.log("Error BpmSender");
        };
        return () => {
            clearInterval(sendDataId);
            try {
                bpmSender.current.close();
            }
            catch (e) {

            }
            try {
                uterusSender.current.close();
            }
            catch (e) {

            }
        };
    }, []);
    return <div>

    </div>;
}
