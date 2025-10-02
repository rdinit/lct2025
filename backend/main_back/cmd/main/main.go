package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/gorilla/websocket"
)

/*
type DataChunk struct {
	data []DataPoint
	mu   sync.RWMutex
}
*/

type DataPoint struct {
	Time  float64
	Value float64
}

type DataManager struct {
	Id                string
	outputConnections []*websocket.Conn
	mu                sync.RWMutex
	data              []DataPoint
	//dataChunks []DataChunk
	//marks []Marks
}

func NewDataManager() *DataManager {
	return &DataManager{
		data: make([]DataPoint, 0),
	}
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
	return &MLHandler{overlap, length, 0, nil, make([]*websocket.Conn, 0)}
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

type DataMerger struct {
	bpm_data        []DataPoint
	uterus_data     []DataPoint
	merged_data     []MergedPoint
	merged_data_str []string
	points_count    int
	eps             float64
	ml_handlers     map[string]*MLHandler
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
		ml_handlers:     make(map[string]*MLHandler),
		mu:              sync.RWMutex{},
	}
}

func (d *DataMerger) mergedPointToCSV(dp MergedPoint) string {
	return fmt.Sprintf("%f,%f\n", dp.Time, dp.Bpm, dp.Uterus)
}

func (d *DataMerger) AppendData(sensor_id string, dp DataPoint) {
	d.mu.Lock()
	defer d.mu.Unlock()
	switch sensor_id {
	case "bpm":
		d.bpm_data = append(d.bpm_data, dp)
	case "uterus":
		d.uterus_data = append(d.uterus_data, dp)
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

	for _, ml_handler := range d.ml_handlers {
		if ml_handler.last_point-ml_handler.overlap+ml_handler.lenght <= int(d.points_count) {
			ml_handler.last_point = d.points_count
			ml_handler.AskML(d.merged_data_str[ml_handler.last_point-ml_handler.overlap : ml_handler.last_point-ml_handler.overlap+ml_handler.lenght])
		}
	}
}

func health_handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "healthy ", rn)
}

var rn int

var dataManagers = make(map[string]*DataManager)

var dataMerger = NewDataMerger()

var upgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool {
	return true
}}

func new_data_handler(w http.ResponseWriter, r *http.Request) {

	sensor_id := r.URL.Query().Get("sensor_id")
	fmt.Println(sensor_id)

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	dataManager := dataManagers[sensor_id]

	c.SetCloseHandler(func(code int, text string) error {
		log.Println("player disconnected:")
		return nil
	})

	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				if _, ok := err.(*websocket.CloseError); ok {
					log.Println("Stop recieving. Disconnect")
					//DisconnectPlayer(pid)
				} else {
					log.Println("reading error :", err)
				}
				break
			}
			if mt != websocket.TextMessage {
				log.Println("no text message, closing connection")
				log.Println("message type: ", mt, "message: ", message, "error: ", err)
				continue
			}
			log.Println("recieved message: ", string(message))
			dataManager.ProcessMessage(message)
		}
	}()
}

func new_reader_handler(w http.ResponseWriter, r *http.Request) {
	sensor_id := r.URL.Query().Get("sensor_id")

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	dataManager := dataManagers[sensor_id]
	dataManager.mu.Lock()
	defer dataManager.mu.Unlock()

	c.SetCloseHandler(func(code int, text string) error {
		dataManager.RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataManager.AddOutputConnection(c)

}

func get_forecast_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.ml_handlers["forecast"].RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataMerger.ml_handlers["forecast"].AddOutputConnection(c)

}

func get_anomaly_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.ml_handlers["anomaly"].RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataMerger.ml_handlers["anomaly"].AddOutputConnection(c)
}
func get_classify_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.ml_handlers["classify"].RemoveOutputConnection(c)
		log.Println("disconnected:")
		return nil
	})

	dataMerger.ml_handlers["classify"].AddOutputConnection(c)
}

func make_forecast_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.ml_handlers["forecast"].ml_connection = nil
		log.Println("disconnected:")
		return nil
	})
	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				if _, ok := err.(*websocket.CloseError); ok {
					log.Println("Stop recieving. Disconnect")

				} else {
					log.Println("reading error :", err)
				}
				break
			}
			if mt != websocket.TextMessage {
				log.Println("no text message, closing connection")
				//log.Println("message type: ", mt, "message: ", message, "error: ", err)
				continue
			}
			//log.Println("recieved message: ", string(message))
			dataMerger.ml_handlers["forecast"].SendUpdate(message)
		}
	}()
}

func make_anomaly_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.ml_handlers["anomaly"].ml_connection = nil
		log.Println("disconnected:")
		return nil
	})
	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				if _, ok := err.(*websocket.CloseError); ok {
					log.Println("Stop recieving. Disconnect")

				} else {
					log.Println("reading error :", err)
				}
				break
			}
			if mt != websocket.TextMessage {
				log.Println("no text message, closing connection")
				//log.Println("message type: ", mt, "message: ", message, "error: ", err)
				continue
			}
			//log.Println("recieved message: ", string(message))
			dataMerger.ml_handlers["anomaly"].SendUpdate(message)
		}
	}()
}

func make_classify_handler(w http.ResponseWriter, r *http.Request) {
	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	c.SetCloseHandler(func(code int, text string) error {
		dataMerger.ml_handlers["classify"].ml_connection = nil
		log.Println("disconnected:")
		return nil
	})
	go func() {
		for {
			mt, message, err := c.ReadMessage()
			if err != nil {
				if _, ok := err.(*websocket.CloseError); ok {
					log.Println("Stop recieving. Disconnect")

				} else {
					log.Println("reading error :", err)
				}
				break
			}
			if mt != websocket.TextMessage {
				log.Println("no text message, closing connection")
				//log.Println("message type: ", mt, "message: ", message, "error: ", err)
				continue
			}
			//log.Println("recieved message: ", string(message))
			dataMerger.ml_handlers["classify"].SendUpdate(message)
		}
	}()
}

func main() {
	rn = rand.Intn(100)
	fmt.Println(rn)

	dataManagers["bpm"] = NewDataManager()
	dataManagers["uterus"] = NewDataManager()

	http.HandleFunc("/health", health_handler)
	http.HandleFunc("/send", new_data_handler)
	http.HandleFunc("/data", new_reader_handler)

	http.HandleFunc("/get_forecast", get_forecast_handler)
	http.HandleFunc("/make_forecast", make_forecast_handler)
	http.HandleFunc("/get_anomaly", get_anomaly_handler)
	http.HandleFunc("/make_anomaly", make_anomaly_handler)
	http.HandleFunc("/get_classify", get_classify_handler)
	http.HandleFunc("/make_classify", make_classify_handler)

	http.HandleFunc("/datasender", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "datasender.html")
	})
	http.HandleFunc("/plot", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "plot.html")
	})

	http.ListenAndServe(":8082", nil)
}
