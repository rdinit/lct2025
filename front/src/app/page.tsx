"use client";

import PlotGraph from "@/components/plotGraph/plotGraph";
import { MAX_PLOT_POINTS } from "@/constants";
import { useEffect, useRef, useState } from "react";

export default function Home () {
    const socketConnection = useRef<WebSocket>(null!);

    const [plotData, setPlotData] = useState<{
        time: number,
        value: number
    }[]>([]);
    useEffect(() => {
        console.log("Opening wss");
        socketConnection.current = new WebSocket("wss://lct.123581321.ru/data?sensor_id=0");
        socketConnection.current.onopen = function () {
            console.log("Connected to WebSocket server");
        };

        socketConnection.current.onmessage = function (event: (Event & { data: string })) {
            const [timeString, valueString] = event.data.split(",");
            const time = parseFloat(timeString);
            const value = parseFloat(valueString);

            setPlotData((state) => {
                if (state.length > MAX_PLOT_POINTS) {
                    state.splice(0, 1);
                }
                return state.concat({
                    time,
                    value
                });
            });
        };

        socketConnection.current.onclose = function () {
            console.log("WebSocket connection closed");
        };

        socketConnection.current.onerror = function (error) {
            console.error("WebSocket error:", error);
        };

        return () => {
            if (socketConnection.current) {
                console.log("Closing wss");

                socketConnection.current.close();
            }
        };
    }, [setPlotData]);

    return (
        <div className="absolute h-[100dvh] w-[100vw]">
            <PlotGraph plotArray={plotData} className="h-[50%] w-full" dotColor="#e69710" lineColor="#e69710" axisColor="#e69710"/>
            <PlotGraph plotArray={plotData} className="h-[50%] w-full"/>
        </div>
    );
}
