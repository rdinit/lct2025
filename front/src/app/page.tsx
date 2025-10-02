"use client";

import DataSender from "@/components/datasender/datasender";
import PlotGraph from "@/components/plotGraph/plotGraph";
import { MAX_PLOT_POINTS } from "@/constants";
import { useEffect, useRef, useState } from "react";

export default function Home () {
    const bpmGetter = useRef<WebSocket>(null!);

    const uterusGetter = useRef<WebSocket>(null!);


    const classifySocketConnection = useRef<WebSocket>(null!);


    const anomalyGetter = useRef<WebSocket>(null!);


    const forecastGetter = useRef<WebSocket>(null!);


    const [isAnomaly, setIsAnomaly] = useState(false);


    const predictbpmData = useRef<{
        time: number;
        value: number;
    }[]>([]);
    const predictuterusData = useRef<{
        time: number;
        value: number;
    }[]>([]);
    const bpmData = useRef<{
        time: number,
        value: number,
        isAnomaly?: boolean
    }[]>([]);
    const uterusData = useRef<{
        time: number,
        value: number,
        isAnomaly?: boolean
    }[]>([]);
    useEffect(() => {
        console.log("Opening wss");
        bpmGetter.current = new WebSocket("wss://14bit.itatmisis.ru/data?sensor_id=bpm");
        uterusGetter.current = new WebSocket("wss://14bit.itatmisis.ru/data?sensor_id=uterus");

        classifySocketConnection.current = new WebSocket("ws://14bit.itatmisis.ru/get_classify");
        forecastGetter.current = new WebSocket("wss://14bit.itatmisis.ru/get_forecast");
        anomalyGetter.current = new WebSocket("wss://14bit.itatmisis.ru/get_anomaly");

        classifySocketConnection.current.onmessage = (event) => {
            console.log(event);
        };
        anomalyGetter.current.onmessage = (event: (Event & { data: string })) => {
            const anomalySplit = event.data.split(",");


            const anomalyTime = Number.parseFloat(anomalySplit[0]);
            const anomalyScore = Number.parseFloat(anomalySplit[anomalySplit.length - 1]);


            console.log(`Anomaly time: ${anomalyTime} ${uterusData.current.length}`);


            if (anomalyScore > 0) {
                return;
            }
            for (let i = 0; i < uterusData.current.length; ++i) {
                if (Math.abs(uterusData.current[i].time - anomalyTime) < 1) {
                    uterusData.current[i].isAnomaly = true;
                }
            }
            for (let i = 0; i < bpmData.current.length; ++i) {
                if (Math.abs(bpmData.current[i].time - anomalyTime) < 1) {
                    bpmData.current[i].isAnomaly = true;
                }
            }
            setIsAnomaly(anomalyScore < 0);
        };
        forecastGetter.current.onmessage = (event: (Event & { data: string })) => {
            const cols = event.data.split("\n");

            console.log(cols.length);
            predictbpmData.current = [];
            for (const col of cols) {
                const [timeString, bpmString, _] = col.split(",");
                if (Number.isNaN(Number.parseFloat(timeString))) {
                    continue;
                }
                predictbpmData.current.push({
                    time: Number.parseFloat(timeString),
                    value: Number.parseFloat(bpmString),
                });
            }
            predictuterusData.current = [];
            for (const col of cols) {
                const [timeString, _, uterusString] = col.split(",");
                if (Number.isNaN(Number.parseFloat(timeString))) {
                    continue;
                }
                predictuterusData.current.push({
                    time: Number.parseFloat(timeString),
                    value: Number.parseFloat(uterusString)
                });
            }
        };

        bpmGetter.current.onmessage = function (event: (Event & { data: string })) {
            const [timeString, valueString] = event.data.split(",");
            const time = parseFloat(timeString);
            const value = parseFloat(valueString);
            if (bpmData.current.length > MAX_PLOT_POINTS) {
                bpmData.current.splice(0, 1);
            }
            bpmData.current = bpmData.current.concat({
                time,
                value
            });
        };
        uterusGetter.current.onmessage = function (event: (Event & { data: string })) {
            const [timeString, valueString] = event.data.split(",");
            const time = parseFloat(timeString);
            const value = parseFloat(valueString);
            if (uterusData.current.length > MAX_PLOT_POINTS) {
                uterusData.current.splice(0, 1);
            }
            uterusData.current = uterusData.current.concat({
                time,
                value
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

            if (uterusGetter.current) {
                uterusGetter.current.close();
            }
            if (forecastGetter.current) {
                forecastGetter.current.close();
            }
        };
    }, []);

    const [data, setData] = useState({
        bpmData: [],
        uterusData: [],
        predictbpmData: [],
        predictuterusData: []
    });

    const [timer, updateTimer] = useState(0);
    useEffect(() => {
        setData({
            bpmData: bpmData.current,
            uterusData: uterusData.current,
            predictbpmData: predictbpmData.current,
            predictuterusData: predictuterusData.current
        });
        const interval = setTimeout(() => {
            console.log("Update");
            updateTimer(timer ? 0 : 1);
        }, 1000);
        return () => {
            clearInterval(interval);
        };
    }, [timer]);

    return (
        <div className="absolute h-[100dvh] w-[100vw]">
            <PlotGraph plotArray={data.bpmData} predictData={data.predictbpmData} className="h-[40%] w-full" dotColor="#e69710" lineColor="#e69710" axisColor="#e69710" title="BPM"/>
            <PlotGraph plotArray={data.uterusData} predictData={data.predictuterusData} className="h-[40%] w-full" title="UTERUS"/>


            <div>
                {isAnomaly ? "Риск гипоксии" : "Всё хорошо"} {isAnomaly}
            </div>

            <DataSender/>


        </div>
    );
}
