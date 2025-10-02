import { useEffect, useRef, useState } from "react";

export type PlotGraphProps = {

    plotArray: {
        time: number,
        value: number,
        isAnomaly?: boolean
    }[];
    predictData: {
        time: number,
        value: number,
        isAnomaly?: boolean
    }[];
    className?: string;

    lineColor?: string;
    axisColor?: string;
    dotColor?: string;
    title: string
    currentTime: number;
};
export default function PlotGraph (props: PlotGraphProps) {
    const { plotArray, predictData, lineColor, axisColor, dotColor, title } = props;
    const [updateTimer, setUpdateTimer] = useState(0);

    useEffect(() => {
        let predictIndex = 0;
        while (predictIndex < predictData.length && predictData.length > 0 && plotArray[plotArray.length - 1].time > predictData[predictIndex].time) {
            predictIndex += 1;
        }

        if (!sensorChartRef.current) {
            return;
        }
        if (!plotArray || plotArray && plotArray.length === 0) {
            return;
        }

        if (boundingBoxRef.current) {
            const boundingBoxWH = boundingBoxRef.current.getBoundingClientRect();

            sensorChartRef.current.width = boundingBoxWH.width;
            sensorChartRef.current.height = boundingBoxWH.height;
        }
        const ctx = sensorChartRef.current.getContext("2d");

        if (!ctx) {
            return;
        }

        const canvasBox = sensorChartRef.current.getBoundingClientRect();

        const canvas = {
            width: canvasBox.width,
            height: canvasBox.height
        };

        ctx.clearRect(
            0, 0, canvas.width, canvas.height
        );
        const padding = 45;
        const chartWidth = canvas.width - padding * 2;
        const chartHeight = canvas.height - padding * 2;


        let minValue = 100000;
        let maxValue = -100;

        for (let i = 0; i < plotArray.length; ++i) {
            minValue = Math.min(minValue, plotArray[i].value);
            maxValue = Math.max(maxValue, plotArray[i].value);
        }


        const minTime = plotArray[0].time;
        const maxTime = plotArray[plotArray.length - 1].time;
        const valueRange = maxValue - minValue;
        const valuePadding = valueRange * 0.1;
        const displayMinValue = minValue;
        const displayMaxValue = maxValue;
        const xScale = chartWidth / (maxTime - minTime || 1) / 1.5;
        const yScale = chartHeight / (displayMaxValue - displayMinValue || 1);

        ctx.strokeStyle = axisColor ? axisColor : "#262626";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(padding, canvas.height - padding);
        ctx.lineTo(canvas.width - padding, canvas.height - padding);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, canvas.height - padding);
        ctx.stroke();

        ctx.fillStyle = "#666";
        ctx.font = "16px Arial";
        ctx.textAlign = "center";

        const timeStep = (maxTime - minTime) / 5;
        for (let i = 0; i <= 5; i++) {
            const time = minTime + i * timeStep;
            const x = padding + (time - minTime) * xScale;

            ctx.beginPath();
            ctx.moveTo(x, canvas.height - padding - 5);
            ctx.lineTo(x, canvas.height - padding + 5);
            ctx.stroke();

            ctx.moveTo(x, canvas.height - padding - 5);
            ctx.lineTo(x, 0);
            ctx.stroke();


            ctx.fillText(`${(time / 60).toFixed(2)}мин`, x, canvas.height - padding + 20);
        }

        ctx.textAlign = "right";
        const valueStep = (displayMaxValue - displayMinValue) / 5;
        for (let i = 0; i <= 5; i++) {
            const value = displayMinValue + i * valueStep;
            const y = canvas.height - padding - (value - displayMinValue) * yScale;

            ctx.beginPath();
            ctx.moveTo(padding - 5, y);
            ctx.lineTo(padding + 5, y);
            ctx.stroke();

            ctx.moveTo(padding - 5, y);
            ctx.lineTo(maxTime * xScale * 1.5 + padding, y);
            ctx.stroke();

            ctx.fillText(value.toFixed(1), padding - 5, y + 4);
        }
        ctx.beginPath();


        for (let i = 0; i < plotArray.length + Math.min(plotArray.length, predictData.length); ++i) {
            const predictDataIndex = i - plotArray.length;


            const isPredict = predictDataIndex >= predictIndex;

            if (i >= plotArray.length && !isPredict) {
                continue;
            }


            const point = isPredict ? predictData[predictDataIndex] : plotArray[i];

            ctx.strokeStyle = lineColor ? lineColor : "#007bff";


            const x = padding + (point.time - minTime) * xScale;
            const y = canvas.height - padding - (point.value - displayMinValue) * yScale;
            ctx.lineWidth = 2;


            if (isPredict) {
                ctx.fillStyle = "#b5ebd0";
                ctx.strokeStyle = "#b5ebd0";
            }
            if (i === 0) {
                ctx.moveTo(x, y);
            }
            else {
                ctx.lineTo(x, y);
            }
            ctx.stroke();
            ctx.fill();
            ctx.beginPath();
            ctx.fillStyle = dotColor ? dotColor : "#007bff";


            if (isPredict) {
                ctx.fillStyle = "#25592d";
                ctx.strokeStyle = "#25592d";
            }


            if (point?.isAnomaly) {
                ctx.fillStyle = "#ff0000";
                ctx.strokeStyle = "#ff0000";
            }


            const radius = point?.isAnomaly ? 3 : 0;
            ctx.arc(
                x, y, radius, 0, 2 * Math.PI
            );
            ctx.fill();
        }


        ctx.fillStyle = "#000";
        ctx.font = "16px Arial";
        ctx.textAlign = "center";
        ctx.fillText(`Сенсор ${title}`, canvas.width / 2, 20);

        const lastPoint = plotArray[plotArray.length - 1];
        ctx.fillStyle = "#d9534f";
        ctx.font = "14px Arial";
        ctx.textAlign = "left";
    }, [axisColor, dotColor, lineColor, plotArray, predictData, title,]);

    useEffect(() => {
        setUpdateTimer(1);
    }, []);
    const sensorChartRef = useRef<HTMLCanvasElement>(null!);

    const boundingBoxRef = useRef<HTMLDivElement>(null!);


    return <div ref={boundingBoxRef} className={props?.className}>
        <canvas ref={sensorChartRef}></canvas>
    </div>;
}
