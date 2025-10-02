package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/websocket"
)

type SensorData struct {
	Time  float64
	Value float64
}

func readCSVFile(filename string) ([]SensorData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		if err == io.EOF {
			return []SensorData{}, nil
		}
		return nil, err
	}

	var data []SensorData
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		if len(record) < 2 {
			continue
		}

		timeSec, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing time in file %s: %v", filename, err)
		}

		value, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing value in file %s: %v", filename, err)
		}

		data = append(data, SensorData{
			Time:  timeSec,
			Value: value,
		})
	}

	return data, nil
}

func getAllCSVFiles(directory string) ([]string, error) {
	files, err := os.ReadDir(directory)
	if err != nil {
		return nil, err
	}

	var csvFiles []string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(strings.ToLower(file.Name()), ".csv") {
			csvFiles = append(csvFiles, filepath.Join(directory, file.Name()))
		}
	}

	sort.Strings(csvFiles)
	return csvFiles, nil
}

func readAllSensorData(directory string) ([]SensorData, error) {
	csvFiles, err := getAllCSVFiles(directory)
	if err != nil {
		return nil, err
	}

	if len(csvFiles) == 0 {
		return nil, fmt.Errorf("no CSV files found in directory: %s", directory)
	}

	log.Printf("Found %d CSV files in %s: %v", len(csvFiles), directory, csvFiles)

	var allData []SensorData
	for _, file := range csvFiles {
		data, err := readCSVFile(file)
		if err != nil {
			return nil, err
		}
		log.Printf("Read %d data points from %s", len(data), file)
		allData = append(allData, data...)
	}

	log.Printf("Total data points from %s: %d", directory, len(allData))
	return allData, nil
}

func connectWebSocket(url string) (*websocket.Conn, error) {
	dialer := websocket.Dialer{}
	conn, _, err := dialer.Dial(url, nil)
	if err != nil {
		return nil, err
	}
	return conn, nil
}

func sendData(conn *websocket.Conn, sensorID string, data []SensorData) error {
	if len(data) == 0 {
		return fmt.Errorf("no data to send for sensor %s", sensorID)
	}

	startTime := time.Now()
	firstTimestamp := data[0].Time

	log.Printf("Starting to send %s data, first timestamp: %f, total points: %d",
		sensorID, firstTimestamp, len(data))

	for i, point := range data {

		elapsedTime := point.Time - firstTimestamp
		targetTime := startTime.Add(time.Duration(elapsedTime * float64(time.Second)))

		now := time.Now()
		if targetTime.After(now) {
			time.Sleep(targetTime.Sub(now))
		}

		dataString := fmt.Sprintf("%f,%f", point.Time, point.Value)

		err := conn.WriteMessage(websocket.TextMessage, []byte(dataString))
		if err != nil {
			return fmt.Errorf("error sending WebSocket message: %v", err)
		}

		if i%100 == 0 {
			log.Printf("Sent %s data: %s", sensorID, dataString)
		}

		if i < len(data)-1 {
			nextPoint := data[i+1]
			timeUntilNext := nextPoint.Time - point.Time
			if timeUntilNext > 0 {
				time.Sleep(time.Duration(timeUntilNext * float64(time.Second)))
			}
		}
	}

	log.Printf("Finished sending %s data", sensorID)
	return nil
}

func main() {

	dataDirectory := "../../../../../data/regular/1"
	webSocketIP := "wss://14bit.itatmisis.ru"

	bpmDir := filepath.Join(dataDirectory, "bpm")
	uterusDir := filepath.Join(dataDirectory, "uterus")

	log.Printf("Reading from directory: %s", bpmDir)
	bpmData, _ := readAllSensorData(bpmDir)

	uterusData, _ := readAllSensorData(uterusDir)

	bpmConn, _ := connectWebSocket(webSocketIP + "/send?sensor_id=bpm")
	defer bpmConn.Close()

	uterusConn, _ := connectWebSocket(webSocketIP + "/send?sensor_id=uterus")
	defer uterusConn.Close()

	done := make(chan bool, 2)
	errors := make(chan error, 2)

	go func() {
		err := sendData(bpmConn, "bpm", bpmData)
		if err != nil {
			errors <- fmt.Errorf("BPM send error: %v", err)
		} else {
			done <- true
		}
	}()

	go func() {
		err := sendData(uterusConn, "uterus", uterusData)
		if err != nil {
			errors <- fmt.Errorf("Uterus send error: %v", err)
		} else {
			done <- true
		}
	}()

	completed := 0
	for completed < 2 {
		select {
		case <-done:
			completed++
		case err := <-errors:
			log.Println(err)
		}
	}
	log.Println("All data sent successfully")
}
