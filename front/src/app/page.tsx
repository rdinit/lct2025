"use client";

import DataSender from "@/components/datasender/datasender";
import PlotGraph from "@/components/plotGraph/plotGraph";
import { MAX_PLOT_POINTS } from "@/constants";
import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

export default function Home () {
    const bpmGetter = useRef<WebSocket>(null!);

    const uterusGetter = useRef<WebSocket>(null!);


    const classifySocketConnection = useRef<WebSocket>(null!);


    const anomalyGetter = useRef<WebSocket>(null!);


    const forecastGetter = useRef<WebSocket>(null!);


    const [hypoxiaScore, setHypoxiaScore] = useState(1);

    const [currentTime, setCurrentTime] = useState(0);

    const [bpmCurrent, setBpmCurrent] = useState(0);
    const [uterusCurrent, setUterusCurrent] = useState(0);
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

        classifySocketConnection.current = new WebSocket("wss://14bit.itatmisis.ru/get_classify");
        forecastGetter.current = new WebSocket("wss://14bit.itatmisis.ru/get_forecast");
        anomalyGetter.current = new WebSocket("wss://14bit.itatmisis.ru/get_anomaly");

        classifySocketConnection.current.onmessage = (event) => {
            const hypoxiaScoreBuf = Number.parseFloat(event.data);
            console.log(`Hypoxia score is: ${hypoxiaScoreBuf}`);

            setHypoxiaScore(hypoxiaScoreBuf);
        };
        anomalyGetter.current.onmessage = (event: (Event & { data: string })) => {
            const data = JSON.parse(event.data);
            for (let i = 0; i < bpmData.current.length; ++i) {
                for (const bpmTimestamp of data.bpm) {
                    if (Math.abs(bpmData.current[i].time - bpmTimestamp) < 0.5) {
                        bpmData.current[i].isAnomaly = true;
                    }
                }
            }
            for (let i = 0; i < uterusData.current.length; ++i) {
                for (const uterusTimestamp of data.uterus) {
                    if (Math.abs(uterusData.current[i].time - uterusTimestamp) < 0.5) {
                        uterusData.current[i].isAnomaly = true;
                    }
                }
            }
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
            setCurrentTime(time);
            setBpmCurrent(value);
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
            setUterusCurrent(value);
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

            <div className="font-alumni flex w-full justify-between px-10 py-12 text-black">
                <div>
                    <div>
                        <span className="text-2xl">Пациент: </span>
                        <span className="text-3xl">Егорова Любовь Ивановна</span>
                    </div>
                    <div>
                        <span className="text-2xl">Возраст {"(лет)"}: </span>
                        <span className="text-3xl">22</span>
                        <span className="ml-5 text-2xl">Неделя беременности: </span>
                        <span className="text-3xl">23</span>
                    </div>
                    <div>
                        <span className="text-2xl">Время наблюдения: </span>
                        <span className="text-3xl">{(currentTime / 60).toFixed(2)}Мин</span>
                    </div>
                    <div>
                        <span className="text-2xl">ЧСС: </span>
                        <span className="inline-block w-[150px] text-3xl">{bpmCurrent.toFixed(1)} уд/мин</span>
                        <span className="ml-5 text-2xl">Сократительная активность: </span>
                        <span className="text-3xl">{uterusCurrent.toFixed(1)}</span>
                    </div>
                    <div className={ clsx("text-2xl", hypoxiaScore < 0.5 ? "text-red-600" : "text-green-600")}>
                        {hypoxiaScore < 0.5 ? "Риск Гипоксии" : "Риск гипоксии отсутствует"}  {(1 - hypoxiaScore).toFixed(2)}
                    </div>

                </div>
            </div>
            <PlotGraph
                currentTime={currentTime}
                plotArray={data.bpmData}
                predictData={data.predictbpmData}
                className="h-[30%] w-full"
                dotColor="#e69710"
                lineColor="#e69710"
                axisColor="#e69710"
                title="ЧСС"
            />
            <PlotGraph
                currentTime={currentTime}
                plotArray={data.uterusData}
                predictData={data.predictuterusData}
                className="h-[30%] w-full"
                title="Сократительная активности"
            />


            {/* <DataSender/> */}


        </div>
    );
}
