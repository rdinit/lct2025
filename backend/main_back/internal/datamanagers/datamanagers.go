package datamanagers

import (
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
)

type DataPoint struct {
	Time  float64
	Value float64
}

type DataManager struct {
	Id                string
	outputConnections []*websocket.Conn
	mu                sync.RWMutex
	data              []DataPoint
	datamerger        *DataMerger
}

func NewDataManager(Id string, datamerger *DataMerger) *DataManager {
	return &DataManager{Id, make([]*websocket.Conn, 0), sync.RWMutex{}, make([]DataPoint, 0), datamerger}
}

func (dm *DataManager) AddOutputConnection(c *websocket.Conn) {
	dm.outputConnections = append(dm.outputConnections, c)
}

func (dm *DataManager) RemoveOutputConnection(c *websocket.Conn) {
	for i, conn := range dm.outputConnections {
		if conn == c {
			dm.outputConnections = append(dm.outputConnections[:i], dm.outputConnections[i+1:]...)
			break
		}
	}
}

func (dm *DataManager) AppendData(dp DataPoint) {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	dm.data = append(dm.data, dp)
}

func (dm *DataManager) SendUpdate(message []byte) {
	if dm.outputConnections == nil {
		log.Println("No output connections")
	}
	for _, conn := range dm.outputConnections {
		conn.WriteMessage(websocket.TextMessage, message)
	}
}

func (dm *DataManager) ProcessMessage(message []byte) {
	splitted := strings.Split(string(message), ",")
	time, _ := strconv.ParseFloat(splitted[0], 64)
	value, _ := strconv.ParseFloat(splitted[1], 64)
	dp := DataPoint{time, value}

	dm.SendUpdate(message)
	dm.datamerger.AppendData(dm.Id, dp)
	dm.AppendData(dp)
}

type MergedPoint struct {
	Time   float64
	Bpm    float64
	Uterus float64
}

type MLHandler struct {
	overlap           int
	lenght            int
	last_point        int
	ml_connection     *websocket.Conn
	outputConnections []*websocket.Conn
}

func (m *MLHandler) AddOutputConnection(c *websocket.Conn) {
	m.outputConnections = append(m.outputConnections, c)
}

func (m *MLHandler) RemoveOutputConnection(c *websocket.Conn) {
	for i, conn := range m.outputConnections {
		if conn == c {
			m.outputConnections = append(m.outputConnections[:i], m.outputConnections[i+1:]...)
			break
		}
	}
}

func NewMLHandler(overlap int, length int) *MLHandler {
	return &MLHandler{overlap, length, -1, nil, make([]*websocket.Conn, 0)}
}

func (m *MLHandler) AskML(messages []string) {
	message := []byte(strings.Join(messages, "\n"))
	m.ml_connection.WriteMessage(websocket.TextMessage, message)
}

func (m *MLHandler) SendUpdate(message []byte) {
	for _, conn := range m.outputConnections {
		conn.WriteMessage(websocket.TextMessage, message)
	}
}

func (m *MLHandler) DisconnectML() {
	m.ml_connection = nil
}

type DataMerger struct {
	bpm_data        []DataPoint
	uterus_data     []DataPoint
	merged_data     []MergedPoint
	merged_data_str []string
	points_count    int
	eps             float64
	Ml_handlers     map[string]*MLHandler
	mu              sync.RWMutex
}

func NewDataMerger() *DataMerger {
	return &DataMerger{
		bpm_data:        make([]DataPoint, 0),
		uterus_data:     make([]DataPoint, 0),
		merged_data:     make([]MergedPoint, 0),
		merged_data_str: make([]string, 0),
		points_count:    0,
		eps:             0.05,
		Ml_handlers:     make(map[string]*MLHandler),
		mu:              sync.RWMutex{},
	}
}

func (d *DataMerger) mergedPointToCSV(dp MergedPoint) string {
	return fmt.Sprintf("%f,%f,%f", dp.Time, dp.Bpm, dp.Uterus)
}

func (d *DataMerger) AppendData(sensor_id string, dp DataPoint) {
	d.mu.Lock()
	defer d.mu.Unlock()
	switch sensor_id {
	case "bpm":
		d.bpm_data = append(d.bpm_data, dp)
	case "uterus":
		d.uterus_data = append(d.uterus_data, dp)
	default:
		log.Println("Unknown sensor id")
		return
	}
	// merge data if both have same timestamp

	for len(d.bpm_data) > 0 && len(d.uterus_data) > 0 {

		if math.Abs(d.bpm_data[0].Time-d.uterus_data[0].Time) < d.eps {
			d.merged_data = append(d.merged_data, MergedPoint{d.bpm_data[0].Time, d.bpm_data[0].Value, d.uterus_data[0].Value})
			d.bpm_data = d.bpm_data[1:]
			d.uterus_data = d.uterus_data[1:]
		} else {
			if d.bpm_data[0].Time < d.uterus_data[0].Time {
				d.merged_data = append(d.merged_data, MergedPoint{d.bpm_data[0].Time, d.bpm_data[0].Value, 0})
				d.bpm_data = d.bpm_data[1:]
			} else {
				d.merged_data = append(d.merged_data, MergedPoint{d.uterus_data[0].Time, 0, d.uterus_data[0].Value})
				d.uterus_data = d.uterus_data[1:]
			}
		}
		d.merged_data_str = append(d.merged_data_str, d.mergedPointToCSV(d.merged_data[len(d.merged_data)-1]))
		d.points_count++

	}

	bplen := len(d.bpm_data)
	utlen := len(d.uterus_data)
	BOUND_VALUE := 3
	if bplen-utlen >= BOUND_VALUE {
		for (bplen - utlen) > BOUND_VALUE {
			fmt.Println(d.bpm_data[0], d.bpm_data[0], 0)
			d.bpm_data = d.bpm_data[1:]
			bplen--
		}
	} else {
		if utlen-bplen >= BOUND_VALUE {
			for (utlen - bplen) > BOUND_VALUE {
				fmt.Println(d.uterus_data[0], 0, d.uterus_data[0])
				d.uterus_data = d.uterus_data[1:]
				utlen--
			}
		}
	}

	for _, ml_handler := range d.Ml_handlers {
		if ml_handler.last_point-ml_handler.overlap+ml_handler.lenght <= int(d.points_count) {
			fmt.Println("points", ml_handler.last_point, ml_handler.overlap, ml_handler.lenght, d.points_count)

			start := ml_handler.last_point - ml_handler.overlap
			end := ml_handler.last_point - ml_handler.overlap + ml_handler.lenght
			fmt.Println("start", start, "end", end)
			ml_handler.AskML(d.merged_data_str[start:end])

			ml_handler.last_point = d.points_count
		}
	}
}
