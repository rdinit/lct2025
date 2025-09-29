import { useEffect, useRef } from "react";

export type PlotGraphProps = {

    plotArray: {
        time: number,
        value: number,
        isAnomaly?: boolean
    }[];
    className?: string;

    lineColor?: string
    axisColor?: string;
    dotColor?: string;
};
export default function PlotGraph (props: PlotGraphProps) {
    const { plotArray, lineColor, axisColor, dotColor } = props;


    useEffect(() => {
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
        const padding = 40;
        const chartWidth = canvas.width - padding * 2;
        const chartHeight = canvas.height - padding * 2;
        const times = plotArray.map((p) => p.time);
        const values = plotArray.map((p) => p.value);

        const minTime = Math.min(...times);
        const maxTime = Math.max(...times);
        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);
        const valueRange = maxValue - minValue;
        const valuePadding = valueRange * 0.1;
        const displayMinValue = minValue - valuePadding;
        const displayMaxValue = maxValue + valuePadding;
        const xScale = chartWidth / (maxTime - minTime || 1);
        const yScale = chartHeight / (displayMaxValue - displayMinValue || 1);

        ctx.strokeStyle = axisColor ? axisColor : "#ccc";
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
        ctx.font = "12px Arial";
        ctx.textAlign = "center";

        const timeStep = (maxTime - minTime) / 5;
        for (let i = 0; i <= 5; i++) {
            const time = minTime + i * timeStep;
            const x = padding + (time - minTime) * xScale;

            ctx.beginPath();
            ctx.moveTo(x, canvas.height - padding - 5);
            ctx.lineTo(x, canvas.height - padding + 5);
            ctx.stroke();

            ctx.fillText(time.toFixed(1), x, canvas.height - padding + 20);
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

            ctx.fillText(value.toFixed(1), padding - 10, y + 4);
        }

        if (plotArray.length > 1) {
            ctx.strokeStyle = lineColor ? lineColor : "#007bff";
            ctx.lineWidth = 2;
            ctx.beginPath();

            plotArray.forEach((point, index) => {
                const x = padding + (point.time - minTime) * xScale;
                const y = canvas.height - padding - (point.value - displayMinValue) * yScale;

                if (index === 0) {
                    ctx.moveTo(x, y);
                }
                else {
                    ctx.lineTo(x, y);
                }
            });

            ctx.stroke();
        }

        ctx.fillStyle = dotColor ? dotColor : "#007bff";
        plotArray.forEach((point) => {
            const x = padding + (point.time - minTime) * xScale;
            const y = canvas.height - padding - (point.value - displayMinValue) * yScale;

            ctx.beginPath();
            ctx.arc(
                x, y, 3, 0, 2 * Math.PI
            );
            ctx.fill();
        });

        ctx.fillStyle = "#000";
        ctx.font = "16px Arial";
        ctx.textAlign = "center";
        ctx.fillText("Sensor 0 - Real-time Data", canvas.width / 2, 20);

        const lastPoint = plotArray[plotArray.length - 1];
        ctx.fillStyle = "#d9534f";
        ctx.font = "14px Arial";
        ctx.textAlign = "left";
        ctx.fillText(`Current: ${lastPoint.value.toFixed(4)}`, padding, 30);
    }, [axisColor, dotColor, lineColor, plotArray]);

    const sensorChartRef = useRef<HTMLCanvasElement>(null!);

    const boundingBoxRef = useRef<HTMLDivElement>(null!);


    return <div ref={boundingBoxRef} className={props?.className}>
        <canvas ref={sensorChartRef}></canvas>
    </div>;
}
