package main

import (
	"fmt"
	"log"
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
	Id                int
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

func health_handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "healthy ", rn)
}

var rn int

var dataManagers = make(map[int]*DataManager)

var data = sync.RWMutex{}

var upgrader = websocket.Upgrader{CheckOrigin: func(r *http.Request) bool {
	return true
}}

func new_data_handler(w http.ResponseWriter, r *http.Request) {

	sensor_id, err := strconv.Atoi(r.URL.Query().Get("sensor_id"))
	fmt.Println(sensor_id)
	if err != nil {
		http.Error(w, "sensor_id must be int", http.StatusBadRequest)
		return
	}

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	dataManager := dataManagers[sensor_id]
	dataManager.mu.Lock()
	defer dataManager.mu.Unlock()

	c.SetCloseHandler(func(code int, text string) error {
		log.Println("player disconnected:")
		return nil
	})

	// Странно работает закрытие на стороне фронта
	// Задержка нескоьлко секунд между нажтием Close на фронте и измением цвета

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
	sensor_id, err := strconv.Atoi(r.URL.Query().Get("sensor_id"))

	if err != nil {
		http.Error(w, "sensor_id must be int", http.StatusBadRequest)
		return
	}

	c, err := upgrader.Upgrade(w, r, nil)

	if err != nil {
		log.Println("upgrade error:", err)
		return
	}

	dataManager := dataManagers[sensor_id]
	dataManager.mu.Lock()
	defer dataManager.mu.Unlock()

	c.SetCloseHandler(func(code int, text string) error {
		log.Println("player disconnected:")
		return nil
	})

	dataManager.AddOutputConnection(c)

}

func main() {
	rn = rand.Intn(100)
	fmt.Println(rn)

	dataManagers[0] = NewDataManager()

	http.HandleFunc("/health", health_handler)
	http.HandleFunc("/send", new_data_handler)
	http.HandleFunc("/data", new_reader_handler)

	http.HandleFunc("/datasender", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "datasender.html")
	})
	http.HandleFunc("/plot", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "plot.html")
	})

	http.ListenAndServe(":8082", nil)
}
