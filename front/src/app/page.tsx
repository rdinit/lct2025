"use client";

import DataSender from "@/components/datasender/datasender";
import PlotGraph from "@/components/plotGraph/plotGraph";
import { MAX_PLOT_POINTS } from "@/constants";
import { useEffect, useRef, useState } from "react";

export default function Home () {
    const bpmGetter = useRef<WebSocket>(null!);

    const uterusGetter = useRef<WebSocket>(null!);


    const classifySocketConnection = useRef<WebSocket>(null!);

    const [bpmData, setBpmData] = useState<{
        time: number,
        value: number
    }[]>([]);
    const [uterusData, setUterusData] = useState<{
        time: number,
        value: number
    }[]>([]);
    useEffect(() => {
        console.log("Opening wss");
        bpmGetter.current = new WebSocket("ws://localhost:8082/data?sensor_id=bpm");
        uterusGetter.current = new WebSocket("ws://localhost:8082/data?sensor_id=uterus");

        classifySocketConnection.current = new WebSocket("ws://localhost:8082/get_anomaly");


        classifySocketConnection.current.onmessage = (event) => {
            console.log(event);
        };

        bpmGetter.current.onmessage = function (event: (Event & { data: string })) {
            const [timeString, valueString] = event.data.split(",");
            const time = parseFloat(timeString);
            const value = parseFloat(valueString);

            setBpmData((state) => {
                if (state.length > MAX_PLOT_POINTS) {
                    state.splice(0, 1);
                }
                return state.concat({
                    time,
                    value
                });
            });
        };
        uterusGetter.current.onmessage = function (event: (Event & { data: string })) {
            const [timeString, valueString] = event.data.split(",");
            const time = parseFloat(timeString);
            const value = parseFloat(valueString);

            setUterusData((state) => {
                if (state.length > MAX_PLOT_POINTS) {
                    state.splice(0, 1);
                }
                return state.concat({
                    time,
                    value
                });
            });
        };

        bpmGetter.current.onclose = function () {
            console.log("WebSocket connection closed");
        };

        bpmGetter.current.onerror = function (error) {
            console.error("WebSocket error:", error);
        };

        return () => {
            if (bpmGetter.current) {
                console.log("Closing wss");

                bpmGetter.current.close();
            }
        };
    }, []);

    return (
        <div className="absolute h-[100dvh] w-[100vw]">
            <PlotGraph plotArray={bpmData} className="h-[50%] w-full" dotColor="#e69710" lineColor="#e69710" axisColor="#e69710" title="BPM"/>
            <PlotGraph plotArray={uterusData} className="h-[50%] w-full" title="UTERUS"/>
            <DataSender/>
        </div>
    );
}
